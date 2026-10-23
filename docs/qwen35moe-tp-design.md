# `--tp 2` for Ornith-1.5-122B (`qwen3_5_moe`): design

Written 2026-10-23, branch `claude/q35moe-tp-design-8151ca` (from `qwen35moe` at `b7b6c83`). **Design spike only**: nothing here was built or
run on a GPU. Every number is either read from the code / docs / container (and cited) or **derived on the CPU** by
[`tools/tp/q35moe_tp_numbers.py`](../tools/tp/q35moe_tp_numbers.py) (byte model checked against the shipped allocation, routing statistics from
the Hessian pass's `routing_counts.json`, 1,003,172 calibration tokens per layer), or an **estimate** and labelled as one. Inputs: `docs/tp.md`,
`docs/lfm2-plan.md` 3.6, `docs/lfm2.md`, `docs/w8a16.md` ("TP=2 outcome"), `docs/trellis-kernel.md` 2.4 / 5.5, `docs/qwen35moe-plan.md`,
`docs/qwen4exp.md` ("Decode performance"), `docs/pp-prefill.md`, `src/model/tp/*`, `src/model/shard_loader.h`,
`src/model/{qwen35moe_*,lfm2_tp_model.*,tp_model.cpp}`, `third_party/libr4d` GDN / attention kernels, `moe_trellis_decode.hip`.

## 0. Answers at a glance

| # | Question | Answer |
|---|---|---|
| 1 | Route through `TpModel`? | **No.** `TpModel` owns a `Model` (the dense Qwen3.5 forward over `Container`); qwen3_5_moe has its own container / stage / model. The refusals in `tp_model.cpp:1298` and `qwen35moe_local_text_model.cpp:75,90-102` are guards, not the obstacle. Follow the precedent of `Lfm2TpModel` / `GemmaTpModel`: a **`Qwen35MoeTpModel`** facade (copy-adapt of `lfm2_tp_model.*`, the smallest) over per-rank `Qwen35MoeModel`s. What is reused unchanged: `TpGroup`, `HostMailboxComm` / `EmulatedComm` / `NoopComm`, `RankWorker`, `tp::RuleFor` + `PlanRows/PlanCols/Gather`, `ShardLoader::TrellisSlice`, `tp_vocab.h`, `AttentionLayer`'s all-reduce hook. |
| 2 | Is the trellis format shardable? | **Yes, exactly, with no re-quantization and no re-packing from the oracle.** The container's `.trellis.w` pair grid is sliced by **byte ranges at load time** (`kTrellisW`, already shipped and run at TP=2 on the Huihui trellis container). Every needed boundary is a multiple of 128 (section 2). The oracle outputs (`oracle-k3`, `oracle-k4`) are **not needed**; the converter writes nothing per rank (tp.md 1.2 non-goal). Only new work: a KB 3 case in `test_tp_shard` (it covers KB 4 / 5), and the expert-stack gather in `Qwen35MoeContainer`. |
| 3 | MoE sharding | **TP-split every expert** (gate and up rows 1024 -> 512 each, down K 1024 -> 512), router and `shared_gate` replicated, shared expert split the same way, **one all-reduce per layer on the combined output** (the LFM2 design, measured to work). Expert parallelism reads **1.27x** the routed bytes on the critical rank at decode (5.1 vs 4.0 experts, section 4) and needs a variable-length grouped kernel; it only wins on prefill launches, which the grouped prefill GEMM removes anyway. |
| 4 | GDN / attention | Value heads 64 -> 32, key heads 16 -> 8 (ratio 4 kept), conv channels 12288 -> 6144; q heads 32 -> 16, **kv head 2 -> 1** (GQA 16 kept, so `r4d_attn_paged_h256_gqa16` applies unchanged: it checks the *ratio* only, `r4d_attn_paged_h256_gqa16.hip:32`, `r4d.hpp:66`). Two kernel-level caveats: the fused-norm GDN decode kernel then runs 32 workgroups (does not speed up), and the gqa16 decode kernel serves `q_len <= 4` only (`.hip:115`), so a 16-row verify needs the prefill / split-KV path. |
| 5 | Memory | **25.997 GiB of weights per rank** (script, section 5) against 25.934 / 25.567 today; KV per rank is the same as the layer split (6 KiB/token); arenas the same. So TP neither helps nor hurts VRAM; device 0 (desktop) stays as tight as today. A replicated DFlash2 drafter (~0.7 GiB w4a16) does not fit device 0 next to the desktop unless measured free VRAM allows; fallback is drafter on rank 1 only. |
| 6 | Plan / gain / effort | Seven phases, **~70-95 h** (~9-12 working days): P0-P5 (~58-78 h) give plain decode + prefill + soak + tuned kernels, P6 (10-16 h) is DFlash2 under TP and waits for the DFlash2 session's merge. Expected plain decode **1.5x (range 1.3-1.7x)** over the layer split: ~62 -> **~93 tok/s (80-105)**. Prefill gets little from TP alone (section 6.4). |

The honest headline: decode gain is real but *estimated*; the first GPU gate (P2, noop mode, one card) turns it into a measurement before the
expensive phases (P3 facade, P4 kernels) are written, the same discipline as tp.md's G3 / G5.

---

## 1. The existing TP implementation and what qwen3_5_moe needs from it

### 1.1 How it shards a layer (docs/tp.md; dense Qwen3.8-27B)

Megatron TP=2, one process, **one rank thread per GPU** (`tp::RankWorker`), a facade thread that broadcasts every call and reconciles the results.
Column-parallel in-projections, row-parallel out-projections, the residual stream and norms **replicated**, one all-reduce on the bf16 `[T, hidden]`
output of each row-parallel site: GDN `out_proj`, attention `o`, MLP `down` = 2 per layer (tp.md 0). The model code is SPMD-aware: it runs on a
rank-local config and calls `core::TpComm::AllReduceSumBf16Rows` itself (`attention_layer.hpp:622` is the attention site).

Shard rules are `tp::RuleFor(base_name, global_config)` (`tp_shard.cpp:322`): per tensor `kReplicate | kRows(segments) | kCols | kRank0Only`; rows
are split **per fused segment** (qkv = q | k | v, gate_up = gate | up), each segment halved, rank r takes half r of each, concatenated. The byte plans
`PlanRows` / `PlanCols` / `Gather` (`tp_shard.cpp`) are CPU-only and tested against the converter's own packers (`tests/model/test_tp_shard.cpp`).

Three modes (`TpOptions::mode`, tp.md 6.4-6.5): **real** (one rank per GPU, `HostMailboxComm`: the host-memory mailbox all-reduce, 7.5 us per 10 KiB, 13 us per 80 KiB, no P2P needed), **emulate** (both shards on ONE device,
two rank threads, `EmulatedComm` = host-synchronous exact add: the byte-exact reference for real mode and the numerics check against TP=1) and **noop** (ONE rank's shard with a no-op all-reduce: per-rank step timing only, and it is what a model that
does not fit one card at TP=1 can still be measured with).

### 1.2 Layouts and what the converter writes

Supported per-rank layouts: bf16, w4a16 (g32 / g64), **trellis** (`Part::kTrellisW`; production TP=2 on the Huihui trellis mix since 2026-09-29, tp.md 807-811, G4),
and w8a16 (LFM2 only). **The converter writes nothing per rank**: tp.md 1.2 lists "a re-converted per-rank container" as a non-goal; every
rank gathers its slice from the one `.r4dx` at load (`ShardLoader`, `shard_loader.h`): one contiguous byte run is uploaded straight from the mmap
(or DirectStorage), several runs go through a reusable host staging buffer. For trellis (`ShardLoader::TrellisSlice`, `shard_loader.h:222-294`):

| tensor | column-parallel (rows) | row-parallel (K) |
|---|---|---|
| `.trellis.w` | one byte run per row segment (pair rows `[n0/32, (n0+n)/32)`) | per pair row, `K/16` tiles `[k0/16, (k0+k)/16)`: N/32 runs |
| `.trellis.suh` (input signs, per K) | replicated (all parts, full K) | cut to the rank's K range |
| `.trellis.svh` (output scales, per N) | cut by the rank's rows | replicated |

### 1.3 What qwen3_5_moe lacks, and the route

`LoadTextModel` dispatches on `DetectArch`; `kQwen35Moe` goes to `LoadQwen35MoeTextModel` and refuses `tp.world != 1` (`tp_model.cpp:1298-1301`,
`qwen35moe_local_text_model.cpp:90-102`; `MakeQwen35MoeModelOptions` refuses `opts.tp.world != 1` at `:75`). Behind the refusal there is no TP code:

| Needed | Exists? | Work |
|---|---|---|
| shard rules for the arch | `RuleFor` falls through to the dense table for `text.layers.*`: the GDN and attention names (`gdn.in_proj_qkv`, `attn.qg`, ...) match, the `moe.*` names are `unknown()` | a `Qwen35Moe` branch next to `LookupLfm2LayerRule` (`tp_shard.cpp:351`): reuse `LookupLayerRule` for gdn / attn, add `moe.router`, `moe.shared_gate` (replicate), `moe.shared.gate|up` (rows `{1024}`), `moe.shared.down` (cols 1024), `moe.experts.{e}.gate_up` (rows `{1024, 1024}`), `.down` (cols 1024) |
| rank-local config | `ModelConfig::Shard` has no knowledge of this arch (it needs a dense `intermediate_size`) | `Qwen35MoeConfig::Shard(world, rank)`: heads, kv heads, GDN key / value heads, `moe_intermediate_size`, `shared_expert_intermediate_size` / world; vocab stays global; checks of section 2.2 |
| sharded container | `Qwen35MoeContainer::Load` has no TP | `Qwen35MoeLoadOptions{tp_world, tp_rank, shared_embed_host}`; p-class + shared expert via `ShardLoader::Linear`, small tensors via `Raw`, **experts via a new `LoadExpertClassSharded`** (section 2.3) |
| rank-local stage | `Qwen35MoeStage` builds `GdnGeometry`, `AttnConfig`, `MoeConfig` from the config (`qwen35moe_stage.cpp:213-249`), so a rank-local `Qwen35MoeConfig` makes them rank-local for free | `TpComm*` in `Qwen35MoeStageOptions`; **3 all-reduce sites** (after `Qwen4ExpGdn::Forward` in `RunGdn`; `AttnConfig::comm` for attention; after `gmoe->Run` / `moe->Forward` in `RunMoe`); vocab-split head |
| facade | `Lfm2TpModel` (35 KB) is the template; it already does host-routed MoE with TP-split experts | `Qwen35MoeTpModel` copy-adapt (docs/lfm2-plan.md open question 3 chose copy over a template; keep that) |

No change is needed to `Qwen4ExpGdn` or `qwen4exp::MoeBlockBase` (shared with qwen4_exp): the stage puts the all-reduce *between* their output and the
residual add, exactly where `RunGdn` / `RunMoe` already hold `blk`. `MoeBlockBase` is parametric (`MoeConfig::inter` / `shared_inter`; its gate_up
convention gate = columns `[0, inter)`, up = `[inter, 2 inter)` is what the per-segment row split produces at `inter = 512`).

Both ranks run the **router on identical bits** (the post-all-reduce residual is bit-identical on both ranks: each element is the same two-term sum), so
they pick the same experts with the same tie-break, and each host-routes on its own readback (the LFM2 argument, `lfm2_tp_model.h` comment).
This is a correctness assumption, not a mechanism: P1's emulate-vs-real byte-equality gate (G6) and a debug-only host compare of the ids cover it.

---

## 2. Shardability of the trellis container

### 2.1 Why slicing is exact

A trellis linear computes `y = svh * H128( W_q · H128( suh * x ) )` where both Hadamards are **128-blocks** (`docs/trellis-kernel.md` 2.3) and `W_q` is stored as
independent tail-biting 16 x 16 tiles (`8 KB` words per tile) in a pair grid (32-row tile pairs, `idx = (((tn>>1)*(K/16) + tk)*2 + (tn&1))*8KB + w`, 2.1).
Neither the codebook decode nor the Hadamards couple tiles or 128-blocks. So

* a **row** range that is a whole number of 128-blocks yields exactly the matching rows of `y` (the output Hadamard acts on whole blocks, `svh` is cut by the same rows);
* a **K** range that is a whole number of 128-blocks yields exactly the partial sum over that range (`suh` cut by the same K range, `svh` replicated, the partial sums add before the output rounding).

The dequantized weights of a rank are bit-for-bit the corresponding slice of the TP=1 dequantized weights. The *only* numeric difference to TP=1 is the one
every row-parallel TP has: partials are rounded to bf16 before the add (tp.md 1.2 "bit-identity with TP=1 is impossible; comparison is KL-based").
The quantization used full-matrix Hessians and LDLQ feedback across the whole K; that happened before packing and is untouched.

**Verdict: slicing only. No re-quantization (hours) and not even a re-pack from `oracle-k3` / `oracle-k4` (minutes).** The shipped container
`ornith-1.5-122b-trellis-mix-kv.r4dx` is sufficient as is; the loader gathers the runs. (Re-packing from the oracle would only be needed if the 128 alignment failed;
it does not.)

### 2.2 Alignment (script table 6; trellis needs rank ranges in whole 128-blocks, rank K % 512 for the tuning rows)

| linear | split | global N x K | rank N (per segment) | rank K | %128 | K %512 |
|---|---|---|---|--:|---|---|
| `gdn.in_proj_qkv` | rows, 3 segments | 12288 x 3072 | 1024 + 1024 + 4096 | 3072 | yes | yes |
| `gdn.in_proj_z` | rows | 8192 x 3072 | 4096 | 3072 | yes | yes |
| `gdn.out_proj` | **cols** | 3072 x 8192 | 3072 | **4096** | yes | yes |
| `attn.qg` (per-head `[q256 \| gate256]` interleave) | rows | 16384 x 3072 | 8192 = 16 whole heads | 3072 | yes | yes |
| `attn.k`, `attn.v` | rows | 512 x 3072 | **256 = one kv head** | 3072 | yes | yes |
| `attn.o` | cols | 3072 x 8192 | 3072 | 4096 (16 heads x 256) | yes | yes |
| expert `gate_up` (two parts, own `suh` each) | rows, 2 segments | 2048 x 3072 | 512 + 512 | 3072 | yes | yes |
| expert `down` | cols | 3072 x 1024 | 3072 | **512** | yes | yes |
| shared gate / up (bf16) | rows | 1024 x 3072 | 512 | 3072 | -- | -- |
| shared down (bf16) | cols | 3072 x 1024 | 3072 | 512 | -- | -- |

All pass. The tightest are the one-kv-head slice (256 rows) and the expert `down` K = 512 (32 k-tiles, which limits the split-K factors; section 3.4). The
`k_descale` / `v_descale` tensors `[2]` split 1 + 1 (`Rows({kv_heads})`, `tp_shard.cpp:90`). `in_proj_a/b` `[64, 3072]` bf16 and `conv1d_weight` `[12288, 4]` use the
dense rules (`{64}` rows, the three-segment qkv rows); `A_log` / `dt_bias` `[64]`.

### 2.3 Experts: the stack gather (the only structurally new loader code)

`LoadExpertClass` (`qwen35moe_container.cpp:365-473`) copies each expert's `.trellis.w` into **one stacked buffer** (`E * w_per` words) because the grouped decode
kernel indexes `all_w + expert * w_per` (`moe_trellis_decode.hip:249`). The sharded version keeps that layout at the rank's size: per expert, `PlanRows` (gate_up: two
runs of 512 rows = 16 pair rows x 192 tiles x 192 B = 590 KB at KB 3) or `PlanCols` (down: 96 runs of 6 KB at KB 3), gathered into the stack at `e * w_per_rank`;
`suh` replicated (gate_up: both parts, 24 KB fp32) or K-cut (down), `svh` row-cut (gate_up) or replicated (down); `trellis_part_n = {512, 512}`. Per-rank
sign-vector replication costs 0.28 GiB (gate_up `suh` on both ranks), included in the 25.997 GiB.

CPU cost: 24.6 k (layer, class, expert) gathers of ~2 runs (gate_up) / 96 small runs (down) per rank. The gate_up runs exceed `DirectLoadScope::kMinBytes` (DirectStorage); the 6 KB down
runs go through the staging buffer. Load time is expected near today's 75-94 s; the risk is the page cache: the 6 KB-on-6 KB-off K-split touches every page of the down class on
both ranks and the file (56.5 GB) does not fit the 64 GB host's cache next to the 2.4 GiB pinned embedding, so load the two ranks layer-by-layer in lockstep (one shared mmap,
one read of each page). Measure at P1; a layer-granular barrier is a few lines.

### 2.4 Tests this requires (CPU, P0)

`test_tp_shard` covers `kTrellisW` at **KB 4 and 5** only (`test_tp_shard.cpp:1012`); Ornith's mix is **KB 3 and KB 4** (25 gate_up + 18 down layers at KB 3,
`allocation_report.md`). Add KB 3 (192 B blocks), the expert shapes `{[2048,3072] rows 512+512, [3072,1024] cols 512}` and the KV-head `{[512,3072] rows 256}` slice, and a
**dequantize check on real tensors**: `tests/kernels/trellis_ref.hpp` has a CPU decoder; for one GDN layer, one attention layer and a few experts of the real container
(read-only mmap), `decode(Gather(plan(rank))) == slice(decode(full))` bit for bit for every linear in section 2.2, both ranks. This is the proof of 2.1 on the shipped bytes.

---

## 3. MoE sharding

### 3.1 The three options

| | **A. TP-split every expert** (recommended) | B. Expert parallel (rank r owns experts `[128r, 128r+128)`) | C. EP routed + TP shared / attention |
|---|---|---|---|
| expert bytes per rank | half of every expert (K-split `down`, row-split `gate_up`) | whole experts, half of them | as B |
| decode, 8 selected | **4.0 expert-equivalents** exactly, every layer | **5.1 on the critical rank** (simulated with the real marginals: 1.27x); the 8 experts split 4/4 in only ~27-30% of layer-tokens, 3/5 in ~44%, 2/6 or worse in ~29% | as B |
| 16-row verify | 49.2 expert-equivalents (98.4 distinct / 2) | 52.6 (1.07x) | as B |
| 256-row chunk | 127.0 | 127.5 (1.00x) | as B |
| all-reduces | 1 per MoE layer on the combined `[T, 3072]` | same | same |
| kernel shapes | N 1024 / K 3072 and N 3072 / K 512 (new instantiations); every rank runs the same `top_k` | **variable `top_k` per rank** (a count-on-device grouped launch, a device-side filter of the router ids) | as B |
| host-routed prefill launches | unchanged (all 256 experts, half-size GEMMs): ~25 k launches per chunk per rank | **halved** (128 experts) | halved |
| sign vectors | replicated gate_up `suh`: +0.28 GiB per rank | none | none |
| precedent | `Lfm2TpModel` (validated bf16 and w8a16 TP=2) | none | none |

Table B's decode numbers are from the script (section 4 table): per-layer Plackett-Luce top-8 draws whose weights are fitted so each expert's inclusion probability equals its measured
routed share. It models neither the router's co-selection structure nor adjacent-token correlation, so distinct-expert counts for n > 1 rows are an **upper bound** (real draft blocks
share experts), and EP imbalance is approximate; but for n = 1 the 5.1 matches the closed form for an unbiased even split (E[max(X, 8-X)] = 5.09 for 8 fair coin flips; the measured shares are skewed
20x between hottest and coldest expert, and a popularity-balanced static assignment did not beat the contiguous split in the simulation).

**Choice: A.** Decode is the headline and the grouped decode kernel is the hot path; A is balanced by construction, keeps every kernel launch shape fixed, and has a validated precedent. B's
only advantage is the prefill launch count, and the planned **grouped prefill MoE GEMM** (plan section 8 item a; one launch per layer, rows sorted by expert) removes the launch-bound regime
for either layout. A hybrid "EP for prefill, TP-split for decode" would need both layouts resident (2x the expert bytes): impossible at 26 GiB per card.

### 3.2 Router, shared expert, sigmoid gate

* **Router** `[256, 3072]` bf16 (1.57 MB per layer, 75 MB model-wide): replicated, computed by both ranks. It is the only MoE weight read in full on both ranks; with the shared expert it is 21% of decode bytes (section 3.3).
* **Shared expert** (bf16, 18.9 MB per layer): gate / up row-split 1024 -> 512, down K-split; the output is a *partial* sum like every expert's, and the sigmoid gate (`shared_gate` `[1, 3072]`, replicated, computed
  identically on both ranks) multiplies the partial: `g * (p0 + p1) = g*p0 + g*p1`, so one all-reduce of the combined `routed + gate * shared` partial is exact up to the bf16 rounding of the two partials.
* **Combine rounding.** The grouped combine kernel already sums the 8 weighted expert rows and the gated shared row in fp32 and rounds once (`GroupedMoeDecodeCombineRouterOrder`). Under TP that single bf16 rounding
  happens on each rank's partial, then the all-reduce adds two bf16 values. So TP adds one extra bf16 rounding per layer relative to TP=1 (tp.md's accepted behaviour) rather than per expert.

### 3.3 Bytes per decode token (script table 5; the 4.70 GB are the weights a token touches, GDN state and lm_head excluded)

| | TP=1 total (the layer split reads the same sum over two cards) | TP=2, one rank | share |
|---|--:|--:|--:|
| GDN / attention mixers | 2.10 GB | 1.05 GB | 45% |
| 8 routed experts | 1.62 GB | 0.82 GB | 34% |
| router + shared expert (bf16) | 0.98 GB | 0.53 GB | 21% |
| **sum** | **4.70 GB** | **2.39 GB** | ratio **0.510** |
| + GDN state read + write (36 layers x 4.2 MB x 2) | 0.30 GB | 0.15 GB | |
| + lm_head (w4a16 g32, 0.444 GiB) | 0.48 GB | 0.24 GB | |
| **per token** | **5.5 GB** | **2.78 GB** | |

At the 435 GB/s the grouped MoE kernel measured (plan 7.1) the layer split's pure-bandwidth floor is 12.6 ms and one TP rank's is 6.4 ms. Note the bf16 router + shared expert are a fifth of the bytes: quantizing the shared
expert (an allocator / converter item, independent of TP) would cut ~0.4 GB per rank.

### 3.4 Kernels

* **Grouped decode MoE** (`moe_trellis_decode.hip:215-260`, `547-621`; `moe_trellis_grouped.h:59-103`): the kernels are templates on `(KB, K, N, n_split, SK, SKG, U, w_per, svh_per, ws_per, tickets_per)` and `INTER`.
  Per rank: the input transform of `x` (K = 3072, `suh` replicated) is **unchanged**; gate_up GEMV becomes `K 3072, N 1024, n_split 512` (grid `(8, SKG, top_k)`); `silu * up` + down input transform `INTER 512`
  (grid 4); down GEMV `K 512, N 3072` (32 k-tiles: `ktw = 32 / (SK * SKG)`, so `SKG in {1, 2, 4, 8}` all leave `steps >= 1`; pick 2 or 4); sort and combine are hidden-generic. This is a mechanical new instantiation set
  (`LaunchGuQ35<KB,SKG>` / `LaunchDownQ35` parametrized on a geometry constant), ~150 lines, plus `Qwen35GroupedMoeDecode`'s `kHidden / kInter` constants and `Supported()` becoming per-rank. The ticket arrays are sized `N/128`
  per expert: 8 and 24.
  **Expected per-layer time** (derived from the measured phase times in plan 7.1: shared 0.045, gate_up 0.060, down 0.057, route 0.0175, transform 0.025 ms, event-bracketed so they sum to 0.20 against a 0.131 ms wall):
  shared and the two GEMVs roughly halve (0.025 / 0.035 / 0.032), route and the gate_up transform are replicated (0.0175, 0.025) -> 0.134 x 0.64 = **~0.086 ms** + the all-reduce (~7 us) = **~0.093 ms** against 0.131: a saving of
  ~1.8 ms per token over 48 layers, ~30% of the layer's time. It is *not* 2x because about a fifth of the layer (route + transform) does not shrink.
* **Host-routed MoE** (prefill, 256-row chunks, captures): `MoeBlockBase` with `inter = 512`; the per-expert `ApplyLinear` shapes `(N 1024, K 3072)` and `(N 3072, K 512)` need rows in
  `gemm_tuning_table_trellis_tp2.inc` / the int8 prefill TP tables or run `FallbackTuning` (correct, untuned; P4).

---

## 4. Routed experts an n-row step touches (script output, mean over the 48 layers)

| rows n | distinct experts / layer | TP-split: expert-equivalents per rank | EP critical rank (contiguous) | EP / TP-split |
|--:|--:|--:|--:|--:|
| 1 | 8.0 | 4.0 | 5.1 | 1.27x |
| 4 | 30.2 | 15.1 | 17.2 | 1.14x |
| 8 | 56.3 | 28.1 | 30.9 | 1.10x |
| **16** (DFlash2 verify) | **98.4** | **49.2** | 52.6 | 1.07x |
| 64 | 210.9 | 105.5 | 108.2 | 1.03x |
| **256** (prefill chunk) | **254.0** | **127.0** | 127.5 | 1.00x |

Bytes read per step (weights only, lm_head and state excluded), TP=1 / one TP rank: **n=1 4.70 / 2.39 GB; n=8 14.44 / 7.31 GB; n=16 22.93 / 11.59 GB; n=256 54.37 / 27.45 GB**. (`plan` 6.1's "~20 GB per 16-row step" is this 22.9 GB;
it is an upper bound because adjacent tokens share experts.) The 16-row verify step is ~4.9x the bytes of a decode token; this drives the DFlash2 break-even in section 6.5.

### 4.1 All-reduce count and payload (hidden 3072, bf16 rows of 6,144 B)

| step | all-reduces | payload each | channel (`tp.md` 6.3.2) | estimated latency each | total per step |
|---|--:|--:|---|--:|--:|
| decode, 1 row | 96 (2 per layer: mixer, MoE) | 6 KiB | ch0 | ~7 us (7.47 us measured at 10 KiB, `tp.md` 1.3) | **~0.7 ms** |
| DFlash2 verify, 16 rows | 96 | 96 KiB | ch0 (<= 174,080 B = 28 rows) | ~14 us (13.3 us measured at 80 KiB) | **~1.4 ms** |
| prefill chunk, 256 rows | 96 sites x 3 sliced calls | 1.5 MiB per site | ch1 (`AllReduceSumBf16Rows` slices to 106 rows) | ~58 us per 640 KiB call (measured at 640 KiB) | **~13 ms** per chunk |

`AllReduceSumBf16Rows` slices by `row_elems` and `MaxAllReduceBytes()` (`tp_comm.hpp:75-92`), so hidden 3072 needs no mailbox change; only `HostMailboxComm::SelfTest` hard-codes `kHidden = 5120` (`tp_comm_host_mailbox.cpp:35`, test rows only).

---

## 5. GDN and attention

**GDN** (36 layers). Per rank: value heads 32, key heads 8 (`GqaRepeats` stays 4), dim 128, conv channels `2*1024 + 4096 = 6144`, `GdnStateManager(max_seqs 1, value_heads 32, 128, 128, conv_dim 6144, kernel 4)`;
state 2.1 MB per layer per rank (4.2 MB at TP=1), 75 MB for 36 layers (plus the same again with `prompt_checkpoint`). The head mapping is by contiguous blocks exactly as the 27B (`hv -> hk = hv / (H / Hg)`,
`r4d_gdn_recurrent_update...hip:100`; the conv kernel's q | k | v channel units, tp.md 4.2): rank r's v heads `[32r, 32r+32)` read k heads `[8r, 8r+8)`, which is what the three-segment `in_proj_qkv` / `conv1d_weight`
split produces. `Qwen4ExpGdn` only checks `K == V == 128`, conv width 4, `H % Hg == 0`, `hidden % 64 == 0` (`qwen4exp_gdn.cpp:73-85`): 32 / 8 passes.
Chunked scan (`Qwen4ExpGdn` prefill, `r4d_gdn_chunk_scan_stable`, `kkt_solve`, `conv_prep2`): grids are per head and per 64-row slice, so the work halves with no change.
Recurrent decode: **the fused-norm kernel launches `(1, H)` workgroups** (`...recurrent_update...hip:195-203`: the row split `vs` is only used without the fused norm), i.e. **64 on one card, 32 per rank at TP=2**: each workgroup does
the same work, so this kernel does **not** speed up (the same observation as the 27B's "24 workgroups per rank", tp.md 1.4). 36 layers x ~10-12 us ~ 0.4 ms per token that stays; an unfused norm with `vs = 2` is a later tuning item (P4).

**Attention** (12 layers). q heads 16 per rank (`qg` rows 8192 = 16 whole `[q256 | gate256]` heads), **kv head 1**, GQA 16, `head_dim 256`, partial rotary 0.25. `r4d_attn_paged_h256_gqa16` selects on
`q_heads / kv_heads == 16` (`r4d.hpp:66`, `.hip:32`), so 16 / 1 qualifies without a code change; the fp8 KV cache is `PagedKvCache(kv_heads 1, 256, ...)`, **6 KiB per token per rank** (12 layers x 512 B), identical to the layer
split's 6 layers x 2 heads x 512 B. The k / v descale per rank is `[1]`. Per-kernel behaviour at one kv head:
* prefill / split-KV prefill: grid `(q_blocks * splits, kv_heads = 1, seqs)`; per-workgroup work is unchanged (16 q heads x 1 kv head), the workgroup count halves. At the 256-row chunk and long context the split-KV law (`PrefillSplitKvSplits`, `attention_layer.hpp:58-69`: "at TP=2 kv_heads per rank is half, so the same law gives each rank twice the segments") restores the fill.
* decode: `decode_splits_g16` is `192 / units` with `units = kv_heads * seqs = 1` -> up to 128 splits (capped at `tiles / 2`), so the machine still fills at long context.
* **verify width**: `r4d_attn_decode_h256_gqa16_fp8kv` serves `q_len * 16 <= 64`, i.e. **`q_len <= 4`** (`.hip:115`, "verify width <= 4 at gqa 16"). A 16-row DFlash2 verify (256 rows of MMA) goes through the **prefill / split-KV path**
  (`prefill_split_kv` with `q_len = 16`), which is correct at any `q_len` (BQ = 24 positions per workgroup). The DFlash2 session is the owner of that decision; TP changes nothing about it (the ratio is the same).

The attention layer's TP hook already exists (`cfg_.comm`, `attention_layer.hpp:622`) and the AttnConfig is built from the rank-local config.

**Vocab-split head.** `lm_head` `[248320, 3072]` w4a16 g32 split 124160 / rank (`RuleFor("lm_head") = Rows({vocab})`, `tp_shard.cpp:383`; 124160 = 7760 tiles); the greedy / sampled / full-logits merges are `tp_vocab.h` (`ArgmaxPair`, host
all-gather of 8 B per row, top-64 + lse summaries, `tp.md` 7). `Qwen35MoeStage::Head` (`:551`) gains `vocab_local` / `vocab_offset`; the three `Kind`s map onto the existing merges as `Lfm2Model` does.

---

## 6. Memory, plan, estimates, risks

### 6.1 Memory per rank

| | GiB | note |
|---|--:|---|
| layers, one rank | 25.775 | script: byte model matches `allocation.json` per layer to 0.00% (51.055 GiB layers + 0.444 lm_head = 51.50 GiB total) |
| lm_head slice | 0.222 | |
| **weights** | **25.997** | today: 25.934 / 25.567 (device 0 / 1) |
| of which replicated by design | 0.07 (+0.28 gate_up `suh`) | norms, router, sign vectors |
| KV (fp8) | 6 KiB / token | 16k ctx 0.09 GiB; 128k 0.75 GiB; 262k 1.5 GiB (same as the layer split) |
| GDN state (+ checkpoint) | 0.075 (+0.075) | |
| arenas / scratch | ~0.9 GiB | the "26.9 GiB per card at 16k ctx" measured today is weights + this |
| comm | ~3.3 MB pinned + 2 x u32[64] | |

Device 1 is headless; **device 0 carries the desktop (3.5-4.6 GiB, `docs/pp-tp2-hybrid.md` row 25) on a 31.86 GiB card**, so rank 0 has 31.86 - 25.997 - 0.9 = ~5.0 GiB for the desktop, KV and anything else: the same squeeze as device 0 in
the layer split, no better, no worse. **DFlash2 drafter** (0.9 B, hidden 3072, 6 layers, 32 / 8 heads x 128, sliding window 2048): bf16 1.8 GB, w4a16 g64 expected ~0.6-0.7 GiB (the 27B's 1.9 B drafter is 1.27 GB w4a16) plus a
~50 MB ring. The tp.md decision (replicate on both ranks) does **not** fit device 0 when the desktop takes 4.6 GiB (slack 0.3 GiB). **Decision:** measure free VRAM on device 0 at P1; replicate if >= 1.5 GiB remain after
weights + arenas at the target `--max-ctx`, else run the drafter on rank 1 only and broadcast the 96 KiB drafter hidden rows through the host (`HostExchange`; ~0.15 ms per round). The taps (layers 1, 7, 14, 20, 26, 32, 39, 45) are
free under TP: the residual is replicated, both ranks hold every tap (in the layer split they live on two cards).

### 6.2 What the layer split has that TP does not

Hybrid (`--tp 2 --pp 2`, docs/pp-tp2-hybrid.md): TP decode with PP prefill stages next to the ranks needs both weight layouts resident. The 27B (9.6 GiB per rank) can; Ornith (26 GiB per card) cannot. So the user picks one mode at load:
**layer split** (today: best prefill TTFT per effort, section 6.4; decode 56-70 tok/s) or **`--tp 2`** (decode).

### 6.3 Phases

Files are paths a phase edits; **[conflict]** marks the files the DFlash2 session is editing now. `qwen35moe_container.*`, `qwen35moe_config.h`, `tp_shard.*`, tests and new files are not on its list (check its branch diff before P1).

| phase | content | files | test / gate | hours |
|---|---|---|---|--:|
| **P0** (CPU, no GPU, **no conflict, can start now**) | `Qwen35MoeConfig::Shard` + invariants; `RuleFor` Qwen35Moe branch; KB 3 + expert-shape + KV-head cases in `test_tp_shard`; `qwen35moe_tp_shard` helpers (expert-stack gather plans); CPU dequantize equivalence on the real container (2.4); free-VRAM probe tool for device 0 | `qwen35moe_config.h`, `tp/tp_shard.cpp`, new `tests/model/qwen35moe/test_q35moe_tp_shard.cpp` (+ `tests/model/CMakeLists.txt`) | **G1**: every tensor name of the real container classified (an unclassified name throws); `decode(rank slice) == slice(decode(full))` for all linears of section 2.2 both ranks; Shard() invariants | 8-10 |
| **P1** (rank stage + sharded container; waits for DFlash2) | `Qwen35MoeLoadOptions` TP fields; `LoadExpertClassSharded`; `Qwen35MoeStageOptions{tp_world, tp_rank, tp_comm}`; 3 all-reduce sites; rank-local `AttnConfig`; vocab-slice `Head`; `Qwen35MoeModelOptions` TP fields, one stage with all layers and no hand-off; `NoopComm` run; **`LoadTiny(layout)`** in the tiny fixture (the existing test cannot load the trellis tiny container: handover note) | `qwen35moe_container.{h,cpp}`, `qwen35moe_stage.{h,cpp}` **[conflict]**, `qwen35moe_model.{h,cpp}` **[conflict]**, `tests/model/qwen35moe/test_model_q35moe_tiny.cpp` | CPU: `Gather`-built stacks equal the TP=1 expert views sliced; **tiny-q35moe, emulate mode (two ranks on one device, `EmulatedComm`) vs the single-device output**: layer-isolated (GDN, attention, router, experts, shared expert) and whole model at T = 1 / 7 / 70 / 300, mean/rms within the bf16 noise the TP=1 test already allows (+ partial-rounding term), greedy 24 of 24 at T = 70; both layouts (bf16, trellis tiny-mix) | 12-16 |
| **P2** (GPU: one card, noop; **the economic gate**) | per-rank step timing: rank 0's shard on one card with `NoopComm`, decode and 16-row and 256-row steps, real container (needs the user's go-ahead for GPU time and `tools\with_lock.ps1`); sweep the grouped GEMV split counts with the existing `GroupedQ35SetSplits` pattern | tool in `tools/`, `tests/model/qwen35moe/` | **G3'**: median per-rank noop decode step <= **0.62 x** the layer split's GPU time (stage 0 + stage 1 + head spans, hipEvent-bracketed in the same session on the same prompt: `Qwen35MoeModel::LastTimings()` is host wall time and a decode step only enqueues, so add an `R4DX_Q35_PROFILE` event pair as plan 8e / `R4DX_Q4_PROFILE` do) = **~7.8 ms** against ~12.6 ms of spans; with all-reduces and the facade that is ~9 ms real. Between 0.62 and 0.75 (~9.5 ms noop, ~11.5 ms real, 1.4x) continue with the pessimistic expectations; above 0.75 stop and re-think before P3 | 6-8 |
| **P3** (real two-GPU facade; waits for DFlash2) | `Qwen35MoeTpModel` (copy-adapt `lfm2_tp_model.{h,cpp}`), `LoadTextModel` / `LoadQwen35MoeTextModel` dispatch (remove the `tp.world` refusals, keep `--pp 2` refused), CLI/server flags (the TP flags already exist), shared pinned embedding, vocab merges, sampling, `Reset`/checkpoint, `TpWarmup`, fault recovery (`-TpFault`); real-mode all-reduce on both cards | new `qwen35moe_tp_model.{h,cpp}`, `qwen35moe_local_text_model.cpp`, `tp_model.cpp:1293-1302`, `model/CMakeLists.txt`, `tools/server/smoke.ps1 -Tp 2` for the arch | **G4'** emulated TP=2 vs TP=1: pooled mean KL(truth\|\|TP2emu) <= KL(TP1) + 0.004 (today 0.124-0.125, so <= 0.129) and top-1 >= 88.0% on the Rung 4 corpus (`r4dx_dump.ps1` extended to `--tp 2`; the 27B moved 0.0385 -> 0.0385); **G6'** real == emulation, every `logprobs.f16` SHA-256-equal from the same binary; **G7'** decode >= **1.35x** the layer split in the same session (~83 tok/s from 61); server smoke (chat, thinking, tool call + follow-up, streaming) clean; `-TpFault` recovers | 14-18 |
| **P4** (kernels + tuning) | per-rank grouped decode instantiations (3.4), per-rank tuning rows (`gemm_tuning_table_trellis_tp2.inc`, the TP int8 tables), optional unfused GDN norm with `vs = 2` | `moe_trellis_decode.hip`, `moe_trellis_grouped.h`, `qwen35moe_grouped_decode.*` **[conflict: the DFlash2 session adds an M = 16 kernel to this file; do after its merge]**, `gemm_tuning_table_*_tp2.inc` | grouped path byte-identical to the host-routed path under TP (the TP=1 equivalence test, `test_model_q35moe_grouped`, extended); phase times vs 3.4 | 10-14 |
| **P5** (soak, prefill) | `tools\tp\soak.ps1` pattern for the arch (5 min, then 60 min, TDR watch: both cards at full load has reset the display before, plan 5); TP-aware grouped prefill MoE GEMM only if the plan's P2 item (a) is built (it is shared and layout-agnostic); `SubmitBounder` units for the 256-row chunk on the display card | tools, tests | **G8'**: 5 and 60 minutes, 0 aborts / divergences / TDRs | 8-12 |
| **P6** (DFlash2 under TP; after the DFlash2 session merges) | drafter placement (6.1), verify with 16 rows per rank (GDN window / rollback is rank-local: the accepted count is identical on both ranks because the logits merge is exact), the 98 KiB verify all-reduces (ch0), top-16 merge across vocab shards | `qwen35moe_tp_model.*`, drafter files | greedy-equal to plain TP decode; acceptance within +-5% of the layer-split run (tp.md G10 pattern) | 10-16 |
| | **total** | | | **~68-94** (P0-P5 ~58-78) |

The parent plan's "P6 `--tp 2`: 2-4 days" is optimistic: even with `Lfm2TpModel` as the template the facade, the sharded expert loader, the soak and the kernel instantiations are ~9-12 working days. P0 is cheap, conflict-free and
de-risks the format question for good; do it first.

### 6.4 Expected gain and why

**Plain decode.** Baseline (measured, plan 9): **56-70 tok/s, planning 62 = 16.1 ms/token** on the layer split. That step is the two stage spans (host-enqueued, GPU-bound) plus ~3-4 ms outside the spans
(host syncs, D2H / H2D hand-off, idle wake-up), a figure *measured* on the sister model with the same structure (`docs/qwen4exp.md` "Decode performance": 18.1 ms of which ~4.1 outside the spans, ~3 ms later) and *assumed* here.
Its bandwidth floor is 12.6 ms (5.5 GB at 435 GB/s), so the kernels are near the floor and the hand-off is the largest non-bandwidth term. TP replaces the hand-off with 96 x ~7 us all-reduces (0.7 ms) and a host merge (~0.3-0.5 ms) and halves each rank's bytes (2.78 GB, 6.4 ms floor).

| bottom-up estimate (ms per token) | bottom-up | pessimistic |
|---|--:|--:|
| MoE layers (48 x 0.093 / 0.115 ms: section 3.4; pessimistic = half-size GEMVs 20% less efficient) | 4.5 | 5.5 |
| mixers + norms + residual: scaling part x 0.55 + non-scaling (GDN recurrent, norms, rope) + 48 all-reduces | 3.5 | 4.5 |
| head (half lm_head + merge) | 0.5 | 0.6 |
| facade round trip, enqueue skew | 0.5 | 0.8 |
| **per token** | **9.0 ms** | **11.4 ms** |
| tok/s | 111 | 88 |
| vs layer split (62) | 1.8x | 1.4x |

Two independent cross-checks pull the planning figure below the bottom-up one: the 27B measured **1.65x** with *no* hand-off in its baseline (tp.md G7: 36.0 -> 59.5 tok/s), and LFM2 (also a bandwidth-bound MoE with TP-split experts and grouped decode, but with no hand-off in its baseline)
got only **1.22x** (171 -> 209 tok/s, `lfm2.md` / `w8a16.md`); a ~3-4 ms hand-off that disappears is worth ~25% on top of either. **Planning figure: 1.5x, range 1.3-1.7x: ~93 tok/s (80-105).** I would
not promise more than 1.5x before P2 measures the per-rank step.

**Prefill.** Nothing here supports a large TP prefill win by itself: the host-routed MoE prefill is launch-bound (plan section 8: ~25 k launches per 256-row chunk, ~8 rows per expert against 64-row tiles), and TP-split keeps the **same launch count per rank** (every
rank runs every expert at half size). Estimated 1.1-1.3x from the halved dense work and GDN scan until the grouped prefill GEMM exists; with it ~1.7-1.9x. All-reduce cost is small (13 ms per chunk against 0.4-0.5 s today). **If TTFT is the goal, the cheaper route is independent of TP:**
the layer split already has two balanced stages, and pipelining prefill chunks across them (`docs/pp-prefill.md`: 1.87x / 1.90x / 1.90x measured on the 27B at 8k / 32k / 64k, stage A / B = 1.06, decode unchanged) is a threading change in `Qwen35MoeModel::RunChunk`, no new numerics. Also build the grouped prefill GEMM (plan P2).

**DFlash2 (16-row verify).** A 16-row step touches **98 of 256 experts per layer** (upper bound) = 22.9 GB at TP=1, **11.6 GB per rank** at TP=2 (section 4), ~4.9x a decode token's bytes. Estimated round: TP ~40 ms (verify 27-33 ms at ~430 GB/s plus 3 ms replicated work, 1.4 ms all-reduce, drafter 3-4 ms,
merge / inject 0.5 ms); layer split ~65 ms (22.9 GB, +3.5 ms hand-off, drafter). **Break-even tokens per round against plain decode: TP 40 ms / (9.0 .. 10.7 ms) = 3.7-4.4; layer split 65 / 16.1 = 4.0.** The 27B's production DFlash2 averaged 2.71 tokens per round; the model card's 5.4-9.2
is for a bf16 target on GB300. For the 3.4-bit Ornith (KL 0.125) 4 is plausible, 2.7 is not enough. At 4 / 6 / 8 tokens per round: TP ~100 / 150 / 200 tok/s, layer split ~62 / 92 / 123 tok/s, i.e. TP is **~1.6x** the layer split with DFlash2 (the 27B measured 1.61x, 73.6 -> 118.8). The decisive unknown is acceptance, not the parallelism;
a shorter verify block (8 rows: 56 distinct experts, 7.3 GB per rank) lowers the break-even and is worth a sweep in the DFlash2 session.

### 6.5 What must wait for the DFlash2 session (and what need not)

That session is editing `qwen35moe_stage.*`, `qwen35moe_model.*` and will add a 16-row verify kernel and GDN rollback (its worktree had no commits when this was written). Order:

* **Now (no conflict):** P0 in full, and this document. `qwen35moe_config.h` is shared, but the edit is additive (`Shard`); `qwen35moe_container.*` is not on the other session's list (the loader work of P1 can be written against the current container on a branch and rebased).
* **After it merges:** P1 (stage hooks, model options), P3 (facade over `Qwen35MoeModel`), P4 (`moe_trellis_decode.hip` will contain its M = 16 kernel; add the TP geometry in the same templated style, do not fork the file), P6.
  The TP stage edits are ~60 lines (a `TpComm*`, three all-reduce calls, the head slice); keep every other TP line in **new files** so the rebase is mechanical.
* **Interfaces the DFlash2 session should keep TP-friendly** (cheap now, expensive later): capture / tap hooks that take a device pointer to the replicated residual (not a stage-internal copy); the GDN state window and rollback expressed over the stage's *own* head count (no `64` / `16` constants); the 16-row MoE kernel keyed on `inter` as a template argument (TP needs 512); no host-side assumption that one stage holds all layers.

### 6.6 Risks (most to least important)

1. **Gain is smaller than estimated** (LFM2: 1.22x). Mitigation: P2 gate before any facade work; the replicated route + gate_up transform (~25% of an MoE layer) and the 32-workgroup GDN decode kernel are the known non-scaling terms; a fused route + transform and an unfused GDN norm are the tuning options.
2. **Router divergence between ranks** (different experts chosen = silent garbage, no hang). Router GEMM + softmax + top-k are deterministic on identical inputs on identical GPUs, but this is an assumption; G6' (real == emulated, byte-equal) and a debug-mode host compare of the ids catch it.
3. **Display GPU under sustained two-card load.** Dual-GPU load has reset the display on this machine (plan 5); TP loads both cards on every token and the MoE prefill chunk is long. Reuse `SubmitBounder` / `submit_layers` (tp.md 2.2), the G8 soak, and the GPU-run approval rule.
4. **VRAM on device 0** (desktop 3.5-4.6 GiB): ~5 GiB of slack, ~0.3 GiB after a replicated drafter; context ceiling ~(slack - drafter) / 6 KiB (~60k tokens at 0.36 GiB, as today). Measure at P1.
5. **Numerics**: one extra bf16 rounding per layer (partials), and the K = 512 expert `down` split-K order. Bounded by G4' on the existing Rung 4 corpus; the engine already adds ~0.003 KL (kl-gate doc), the 27B moved by < 0.0001 under TP.
6. **Load time / page cache** (2.3). Measure at P1; layer-lockstep loading if needed.
7. **DFlash2 does not pay** on this MoE below ~4 accepted tokens per round (6.4), whatever the parallelism.
8. **Real routing is not what the simulation assumes** (correlated top-8, correlated adjacent tokens): EP-vs-TP numbers could move a little; the decision (A) does not depend on them, it rests on balance by construction.

### 6.7 Decisions for the user

1. **Mode choice.** `--tp 2` and the layer split are exclusive at 26 GiB per card; keep the layer split as the default until P3's gates pass, then make `--tp 2` opt-in. (TTFT: also consider the cheap chunk-overlap in the layer split.)
2. **Start P0 now** (CPU only, ~8-10 h, no conflict)? It settles the format question for good.
3. **GPU time for P2** (one card, noop mode, ~1 h) after the other session's GPU work: it is the go / no-go for ~45 h of P3-P5.
4. Shared-expert quantization (bf16 -> trellis or w8a16; ~0.4 GB per rank, ~0.9 GB per token on one card, 20% of decode bytes) is a larger and TP-independent decode win; worth its own track.

---

## Appendix A. Script output (`python -I tools\tp\q35moe_tp_numbers.py`, seed 1, ~40 s)

```
### 1. Byte model of the container (TP=1) vs allocation.json
layers 51.055 GiB + lm_head (w4a16 g32) 0.444 GiB; allocation.json total 51.501 GiB (layer sum 51.057); worst per-layer deviation 0.00%

### 2. Weights per rank, every linear split (experts TP-split), TP=2
| layers, TP=1 | 51.055 |   | layers, one rank at TP=2 | 25.775 |   | replicated by design (norms + router) | 0.071 |
| lm_head one rank (vocab split) | 0.222 |   | weights one rank | 25.997 |   | gate_up suh replicated on both ranks | 0.281 |

### 3. Routing statistics (1,003,172 calibration tokens per layer)
mean inclusion probability 0.03125 (uniform = 0.03125); per-layer max/min expert share median 20.3x; hottest expert in any layer 0.215

### 4. Routed experts read per layer by an n-row step (mean over the 48 layers)
| rows n | distinct | TP-split per rank | EP contiguous | EP interleaved | EP balanced | worst EP / TP-split |
|  1 |   8.0 |   4.0 |   5.1 |   5.1 |   5.1 | 1.27x |
|  4 |  30.2 |  15.1 |  17.2 |  17.2 |  17.2 | 1.14x |
|  8 |  56.3 |  28.1 |  30.9 |  30.8 |  30.7 | 1.10x |
| 16 |  98.4 |  49.2 |  52.6 |  52.3 |  52.2 | 1.07x |
| 64 | 210.9 | 105.5 | 108.2 | 107.9 | 107.7 | 1.03x |
|256 | 254.0 | 127.0 | 127.5 | 127.4 | 127.4 | 1.00x |

### 5. Bytes read per step per rank (weights; lm_head excluded)
| n = 1   |  4.70 GB |  2.39 GB | 0.510 | EP-contiguous critical rank  2.61 GB |
| n = 8   | 14.44 GB |  7.31 GB | 0.506 |  7.82 GB |
| n = 16  | 22.93 GB | 11.59 GB | 0.506 | 12.18 GB |
| n = 256 | 54.37 GB | 27.45 GB | 0.505 | 27.33 GB |
```

(Section 6 of the script's output, the alignment table, is section 2.2 above. Per-token breakdown in 3.3: mixers 2.096 / 1.049, 8 routed experts 1.617 / 0.816, router + shared 0.982 / 0.529 GB, TP=1 / one rank.)
