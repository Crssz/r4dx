# Megakernels for LLM decode: do they apply to r4dx qwen4_exp on 2x R9700 (gfx1201, Windows)?

Scope: literature + source review (primary sources only: arXiv papers, the authors' own blog posts and
repos, AMD/ROCm/LLVM docs, the local ROCm 7.15 SDK headers), then a read of this repo's own decode
measurements. **No GPU job was run.** Tags: **documented** = stated by a cited source; **measured** = a
number from this repo's docs (file:line); **[INFERENCE]** = my reasoning, not stated by any source.
Verdict is in section 5.

## 1. What a megakernel is, and what the published ones report

A megakernel (persistent kernel) runs the whole forward pass as one launch whose workgroups stay
resident and pull work from an in-kernel schedule, instead of one launch per operator. Claimed gains
come from three eliminated costs: per-kernel launch/teardown, the grid-wide implicit barrier at every
kernel boundary (stragglers, ramp, drain), and the fact that the next op's weights cannot be fetched
while the previous op finishes (Hazy blog, "Separate Kernels Kill the Vibe":
https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles).

| System | Result as reported | Baseline / GPU / batch | Source |
|---|---|---|---|
| Hazy Llama-3.2-1B (ThunderKittens, hand-written) | 78% of H100 bandwidth; ~2.5x vs vLLM, >1.5x vs SGLang; B200 <680 us/forward, >1.5x SGLang, >3.5x vLLM. Prior engines use "at most 50%" of H100 BW on this workload | vLLM/SGLang with CUDA graphs + torch.compile; H100 and B200; batch 1, 32-token prompt, 128 generated, no speculation | [1] |
| Hazy B200 breakdown (600 us) | 250 us store/await-consistency/load activations; 200 us matvec; 30 us waiting for weights; 40 us warp-sync; 80 us setup. Each instruction pays ~2 load + 2 store latencies (~500 ns each) | same | [1] |
| MPK / Mirage Persistent Kernel (compiler + runtime) | 1.0-1.7x lower per-token latency at batch 1; e.g. Qwen3-8B on A100 14.5 ms -> 12.5 ms vs a ~10 ms bound (16 GB at 1.6 TB/s) | vLLM, SGLang; A100/H100/B200; 5 models incl. Qwen3-30B-A3B MoE; max batch 1-16, prompt 64, 1024 generated | [2] |
| MPK ablations | Qwen3-8B = 293 launches/token; on B200 3.8 us each eager (1.1 ms/token), 0.8 us each in a CUDA graph (0.2 ms/token); cross-task pipelining 1.2-1.3x on the final linear layer (B200); 1.1-1.4x over vLLM/SGLang on 8xH100 TP | same | [2] |
| Fleet (AMD Research, MPK ported to HIP) | bs=1 Qwen3-8B bf16: vLLM 10.51 ms -> MPK-port 7.83 ms -> Fleet 6.82 ms (1.54x over vLLM). Fleet's gain over the MPK port at bs<=16 is "reduced scheduler-to-worker communication overhead", not cache reuse | **vLLM v0.17.2 on ROCm with `--enforce-eager`** (no graphs); MI350X (CDNA, 8 XCDs); decode TPOT | [3] |
| ForgeMegakernel (agent-generated, independent authors) | MBU 50.5-85.9%; geomean 1.21x over SGLang 0.5.18 kernel-level, 1.03-1.28x end-to-end inside SGLang (Qwen3-8B b=1: 5601 vs 6009 us/token) | SGLang with CUDA-graph replay; H100; 0.6B-13B | [4] |
| Forge measurement of MPK | MPK reached only 0.47-0.91x (geomean 0.65) of SGLang's bandwidth utilization on their five cells; a hand-written one reached 74.5% | same H100 | [4] |
| MegaQwen (hobby, Qwen3-0.6B, cooperative `grid.sync`) | 531 tok/s vs TRT-LLM 355, vLLM/SGLang 107, HF 136 -- but only 5% of memory BW; 140+ `grid.sync` per token at ~0.7 us | RTX 3090, bs 1, short context | [5] |
| qwen_megakernel (descendant, non-cooperative persistent kernel) | 1000 tok/s, 71% BW; 6 barriers/layer, one full-grid atomic barrier ~2.2 us, `cg::grid.sync` ~3 us on this GPU | RTX 5090, Qwen3-0.6B bf16; "8.4x" in its README is vs HF eager | [6][7] |

What Forge's decomposition says about *where* the gain comes from (SGLang with graph replay, H100) [4,
section 5.2.3]: graph replay removes host cost but "still starts and retires a grid per kernel"; that
costs 301-653 us/token over 65-141 GEMV launches; launch-boundary waiting 333-851 us vs 227-313 us for
the megakernel; and 111-239 launches per token carry almost no weight bytes (RMSNorm, RoPE, KV store,
combine), costing 225-455 us = **17-21% of the step**. Baselines in all of the above are unfused
library-kernel engines with 280-448 launches/token, not custom-fused decode.

## 2. Mechanisms and AMD/HIP prerequisites

- **Instruction interpreter + static schedule.** Each SM gets a precomputed instruction list; every
  instruction is one template (load/compute/store); schedules are reused for hundreds of passes [1].
  MPK instead splits CUs into workers (own task queue, circular buffer in global memory) and scheduler
  warps (event queues, `atomicAdd`), with JIT dispatch for data-dependent ops (attention) and AOT
  pre-enqueue for the rest; a task descriptor is 352 B [2].
- **Dependencies = counters in global memory, not grid barriers.** Counters start at 0, producers
  increment, consumers wait for a target; Hazy splits the MLP intermediate into 4 chunks with a counter
  each so down-proj starts per chunk [1]. Forge makes "zero grid-wide barriers, >=4L counters" a
  gated milestone; an agent that kept 225 grid barriers per step stalled at 34.2% MBU, barely above
  SGLang's 31% on that cell [4].
- **Shared-memory paging + cross-op prefetch.** Hazy pages 213 KB of H100 smem into 13 x 16 KiB pages
  that instructions request/release, so the next instruction can issue weight loads before the
  previous finishes [1]; MPK splits a task into pre-load and compute phases and overlaps pre-load with
  the previous task [2]. Because weights do not depend on activations, loads can be issued before the
  dependency counter is satisfied. qwen_megakernel does the same with `prefetch.global.L2` issued by
  idle blocks during attention (+10%, 905 -> 1000 tok/s), and measured prefetching *into contended*
  bandwidth as -8.7% [7].
- **Register budget is fixed per kernel.** MPK: smem is paged per task, but "per-thread register usage
  is fixed ... at the maximum required across task types" [2, section 7].
- **Co-residency.** Spin-waiting on another workgroup is only safe if that workgroup is already
  resident. HIP exposes `hipLaunchCooperativeKernel` / `hipDeviceAttributeCooperativeLaunch`
  (declared in the local `C:\opt\rocm\include\hip\hip_runtime_api.h`; usage documented at [8]).
  **AMD GPUs "do not support independent thread scheduling"; code relying on it can deadlock** [8].
- **`grid.sync()` on gfx12.** `cg::grid_group::sync()` -> `__ockl_grid_sync()` (local
  `amd_detail/hip_cooperative_groups_helper.h`). In ROCm-Device-Libs the hardware GWS barrier is
  skipped for ISA >= 11000 (`AVOID_GWS`); it is then a device-scope atomic counter polled with
  `s_sleep(1)`, bracketed by workgroup/agent release+acquire fences, and it reads a runtime-filled
  hidden kernel argument (`mg_info`) that exists only for cooperative launches [9]. So on gfx1201 a
  cooperative `grid.sync` is "just" an atomic barrier plus launch-time residency check -- nothing the
  hardware does for free.
- **Does cooperative launch work on Windows HIP SDK / gfx1201?** Not stated in any primary doc I could
  find (HIP SDK for Windows is documented only as "a subset of the ROCm platform" [10]; repo notes the
  same gap: `docs/research/r9700-arch.md:386-391,470-471`; `docs/r9700.md:750` explicitly never used it).
  **Unverified.** The fallback needs no runtime support: a normal launch with grid <= resident capacity
  and hand-rolled counters, which is what [7] switched to (and says cooperative launch "imposes
  constraints on block count and carries per-barrier overhead") and what this repo already ships
  (below). Residency of a non-cooperative grid is not guaranteed by the API [INFERENCE: holds in practice
  when grid <= CUs x occupancy and nothing else is running].
- **Memory ordering on gfx12.** LLVM's gfx12 memory model maps load/store scopes to
  `SCOPE_CU / SE / DEV / SYS` bits [11]. The repo already uses agent-scope release/acquire fences around
  `atomicAdd` tickets for split-group reduction (`docs/trellis-kernel.md:686-690`, "nobody ever waits")
  and system-scope flags in pinned host memory between two R9700s with 12.28M bit-exact exchanges
  (`docs/tp.md:94,1051-1053`), plus an ISA check of the emitted fences. Cross-workgroup ordering is
  therefore proven usable here. Two gfx12 traps from this repo: a plain `__syncthreads()` emits
  `global_inv scope:SCOPE_SE` on both sides, discarding L0/L1 lines (`r9700-kernel-lessons.md:155-162`,
  uses `lds_barrier()` instead) -- a megakernel with many barriers must follow that; and
  `llvm.prefetch` is "implemented on gfx1250, ignored on earlier targets" [11], with no async-LDS or
  tensor-load on gfx1201 (`r9700-arch.md:186-188`), so Hazy/MPK-style asynchronous smem paging has no
  hardware equivalent -- next-op prefetch would be early register loads or dummy loads [INFERENCE].
- **Chiplet note.** Fleet's synchronization design (two-level counters, `buffer_wbl2`) exists because
  MI350's eight XCDs have private non-coherent L2s [3]. R9700 is one die with one 8 MiB L2
  (`r9700-arch.md:30`); device-scope sync should be cheaper than on MI350 [INFERENCE], but no number
  exists for gfx1201.

## 3. Applicability to this workload

### 3.1 What is already measured here

- 61.0 tok/s = 16.39 ms/token after the HC + QSA fusions (`docs/qwen4exp.md:336`). Decode is GPU-bound:
  host enqueues a stage in 1-1.3 ms vs 6-8 ms GPU; ~0.25 ms/layer, ~39 small dependent kernels at
  ~6-7 us; ~4 GB weights/token; big kernels at 345-485 GB/s = 65-75% of 640 GB/s (`:256-258,323`).
  Achievable on this card is 550-604 GB/s (`docs/trellis-kernel.md:738-739`, C5/C8), so those kernels
  sit at ~75-85% of *achievable*.
- **Graphs for every layer + head: no gain beyond noise** (59.2/59.1 vs 59.6/59.2 tok/s), reverted
  (`:261-267`); `docs/tp.md:96` measured +0.4% on a separate branch. Empty-kernel launch 1.204 us on a
  stream vs 0.712 us in a graph (`docs/r9700.md:70-71`).
- Launches/token 1922 -> ~1536 with the HC fusion; removing 388 launches from a graph saved only
  0.1-0.2 ms (~0.3-0.5 us each) (`:300,325`). Merging route+sort took the kernel from 14.4 to 8.3 us in
  isolation (predicted -0.29 ms/token over 48 layers) and moved the step by 0.01 ms (`:357-359`).
- GPU spans 7.5 + 5.7 + 0.65 = 13.85 ms, ~4.1 ms outside spans (host syncs, D2H/H2D hand-off) with
  ~1 ms event overhead (`:244-254`); remove-one costs sum to 13.9 of 15.3 ms (`:299`). `hipDeviceCanAccessPeer`
  is 0 both ways; host hand-off 0.3-0.66 ms (`:268-272`).

### 3.2 Where the 13.9 ms goes (grouping of the remove-one table, `:302-321`)

| group | labels (ms/token) | ms | character |
|---|---|---:|---|
| A. weight-streaming linears | hc.gemm1 1.49, hc.up_mix 1.30, moe gate_up 1.45 / down 0.83, shared x4 1.39, gdn qkv 0.90 / z 0.74 / out 0.66, lm_head 0.73, qsa qg 0.39 / o 0.23 | 10.1 | ~4 GB at ~395 GB/s average [INFERENCE: sum/bytes] |
| B. latency-chain, little weight traffic | hc.norm 0.73, gdn.core 0.87, qsa.core 0.67, moe.route 0.49, gdn in_proj_ab 0.16, ple 0.14, moe gate_up_in/combine/silu_down_in 0.31, rest <=0.09 each (~0.4) | 3.8 | 2-17 us kernels at 4-48 workgroups |

### 3.3 Honest upper bound

1. **Launch-boundary term, hard ceiling ~1.1 ms (6.7%).** 1536 dependent launches x 0.712 us graph
   cost (`docs/r9700.md:71`). That is the most a megakernel can remove *if its own dependency edge cost
   nothing*. It does not: in-kernel edges measured on NVIDIA cost 0.7 us (3090 `grid.sync`) to ~2.2-3 us
   (5090 barrier / cooperative sync) [5][7], Hazy's per-instruction check-ready/store/mark costs ~2 us
   on B200 [1], and the repo's own estimate for a three-round-trip split tail is 0.5-1.5 us
   (`docs/trellis-kernel.md:712-715`). Net edge saving is ~0-0.5 ms (<=3%) [INFERENCE], matching the
   +2.4% (0.4 ms) the two real fusions gave.
2. **Published gains are against a different baseline.** vLLM/SGLang run at ~31-69% of bandwidth with
   280-448 unfused launches [1][2][4]; MPK moves A100 Qwen3-8B from 69% (14.5 vs 10 ms bound) to 80%
   (12.5 ms), i.e. +16% [2]. r4dx big kernels are already at 65-75% of peak with the small ops fused and
   graphs on. The gain classes that remain for a megakernel are (a) cross-op weight-fetch overlap
   (ramp/drain of ~39 kernels/layer; MPK measured 1.2-1.3x on one linear [2]) and (b) overlapping group B
   latency chains with group A fetches. Neither has been measured on this GPU.
3. **Paper ceiling.** 4 GB at 550-604 GB/s is 6.6-7.3 ms; a Hazy-class 75% of peak (480 GB/s) is 8.3 ms.
   Replacing the 13.9 ms GPU span with 8.3 ms gives 16.39 -> ~10.8 ms = ~92 tok/s (1.5x) [INFERENCE,
   arithmetic only]. This is not a megakernel-specific number: closing A from ~395 to 480 GB/s by
   per-kernel work recovers ~1.8 ms of it without any fusion.
4. **Realistic envelope** for a full megakernel: capture perhaps 1/4-1/2 of group B's 3.8 ms plus a
   small part of A's remaining gap, i.e. **~0.5-2 ms = 3-12% (61 -> ~63-69 tok/s)** [INFERENCE; the
   route+sort result in 3.1 shows small-kernel latency is partly hidden already].
5. **Not removable by a per-stage megakernel:** the ~2.5-4 ms outside the GPU spans (token readback,
   PLE host rows, host sync) and the stage hand-off; no P2P (`docs/qwen4exp.md:271-272`).

### 3.4 Hard parts specific to this model

| Item | Problem | Note |
|---|---|---|
| MoE routing | Expert ids, counts and weight addresses are known only after route+sort, so the cross-op weight prefetch that makes megakernels pay cannot start for gate_up/down; MPK needs a hybrid static/dynamic balancer and a fused gather-GEMM for it (gather was up to 11% of SGLang MoE time at bs1 on Qwen3-30B-A3B) [2, 6.4]. r4dx already has on-GPU sort + grouped GEMV with shapes baked in (hidden 2560, expert 640, <=16 experts; `qwen4exp.md:113-117`) | prefetch benefit does not apply to ~2.3 ms of group A [INFERENCE] |
| GDN decode | `gdn.core` runs 48 workgroups and a head cannot be split because of the fused gated norm (`:365`) -> up to 16 of 64 CUs idle for 0.87 ms if one WG per CU; the one place a persistent design could genuinely help is using idle CUs to prefetch out_proj weights | [INFERENCE] |
| QSA | partial attention is 2 kv heads x 4 WGs at short context (`:350-354`); the reverted graph experiment already had to make its geometry position-independent (`:261-263`) | tiny parallelism, same overlap opportunity as GDN |
| Trellis decode ALU | ALU need/budget is 0.62-0.90 of the memory time per fragment (`trellis-kernel.md:741`); extra live registers for prefetch or spin state eat into an already ALU-co-limited kernel | prefetch into VGPRs competes with decode state |
| Registers/LDS | one kernel => one VGPR count = max over fused ops (MPK [2]); libr4d shows sensitivity: GDN scan at 242 VGPRs spills if one more is live, forcing 10 waves/SIMD on kkt_solve is 1.57x slower (`r9700-kernel-lessons.md:98-99`) | decode ops are lighter [INFERENCE] but unmeasured |
| 2-GPU split | One megakernel per card; stage hand-off stays a host bounce (0.3-0.66 ms). Existing kernel-flag channel between the two cards is 4.5-5.6 us per 10 KiB exchange (`docs/tp.md:2686-2691`) | see 4.3 |
| Windows / display GPU | Device 0 drives the desktop with `ComputePreemptionSupported=0`; TDR is 2 s. A ~8 ms persistent kernel is fine, but back-to-back saturating work TDR'd a stress run after ~200 s (`docs/tp.md:95,2670-2680`). Every spin needs a timeout/abort like the TP kernels (500 ms) | operational risk, not a blocker |
| Build/maintenance | Hand-written megakernels target one model and cost "weeks of specialist effort" [4]; MPK is ~44K C++ / 42K CUDA lines and CUDA/NVSHMEM-only [2]; Fleet's HIP port is CDNA [3]. A 20-op single TU also hides mid-state errors [4] and the repo's bit-identical A/B gate | no RDNA4/Windows reference implementation exists that I found |

## 4. Cheaper alternatives (ordered by evidence, not by glamour)

1. **Raise group A streaming rate per kernel** (no megakernel). 395 -> 480 GB/s is ~1.8 ms (11%);
   `qwen4exp.md:365` calls ~450 GB/s "near the floor" while `trellis-kernel.md:738-739` budgets 550-604
   GB/s as the achievable rate -- reconcile that first. Candidates already named in-repo: bf16 HC
   matrices (1.25 GB, `:324`), k/v at 335-406 GB/s vs 465-478 (`trellis-kernel.md:1344`).
2. **Horizontal/vertical fusion with the existing last-arriver ticket scheme** inside the MoE block
   (route+sort+quant, silu+down-quant, combine: 0.8 ms of cost over four 1.7-10 us kernels) and
   `hc.norm` into `HcPrepGemm1Kernel` (0.73 ms, listed at `:364`). Expect <=0.3 ms
   total given the route+sort precedent [INFERENCE]; it is the same class of change that already paid
   +2.4%.
3. **Device-side stage hand-off** using the existing flag kernels (pinned-host mailbox, 4.5-5.6 us)
   instead of D2H + host sync + H2D (0.3-0.66 ms). Saves <=0.3-0.6 ms (2-4%) [INFERENCE]; PLE rows and
   the sampled-token readback still need the host each step (`qwen4exp.md:273-274`), so the remaining
   ~2 ms outside spans is untouched.
4. **Persistent kernel for the MoE block only** (7 launches, ~3.1 ms/token = ~64 us/layer, plus the 4
   shared-expert GEMVs if co-scheduled): the one sub-graph with many short dependent stages. Build only
   if experiment E2 below passes.

## 5. Verdict and de-risking experiment

**Full-model decode megakernel: NO-GO now.** The mechanisms transfer (counters, per-task dependency,
cross-op prefetch) but: (a) the launch/graph overheads that justified published megakernels are
already measured as ~zero here (graphs +0.4% to 0%, 0.3-0.5 us per removed launch); (b) the paper
ceiling (1.5x) is dominated by per-kernel bandwidth, which is reachable without a megakernel;
(c) the realistic megakernel-specific envelope is 3-12% [INFERENCE] against a rewrite of ~20
heterogeneous kernels, no RDNA4/Windows reference, no async-LDS/prefetch instruction, bit-identity
gates, and unverified cooperative-launch support; (d) MPK itself was measured below SGLang on some
cells by an independent group [4]. **GO only for a one-day measurement (no model code):**

- **E0 (host-visible, ~1 h).** On both cards: `hipDeviceGetAttribute(hipDeviceAttributeCooperativeLaunch)`;
  a `hipLaunchCooperativeKernel` with `grid.sync()` x1000 at 64 and 128 workgroups; the same with a
  normal launch + atomic counter barrier (generation counter, agent-scope fences, workgroup-level
  `lds_barrier`, `s_sleep` backoff, 500 ms timeout). Report us/barrier. **Gate:** <= ~1 us/barrier
  (break-even vs the measured 0.3-0.7 us graph edge); 2-3 us (the NVIDIA numbers [5][7]) means barrier-
  style megakernels cannot win and only per-task counters remain viable.
- **E2 (bubble upper bound, ~1 day).** Take the MoE block (route, sort, gate_up_in, gate_up GEMV, silu_down_in,
  down GEMV, combine) over 48 distinct layers' weights (cycle weights > 64 MiB so Infinity Cache cannot
  flatter it, `r9700-kernel-lessons.md:229-231`). Measure us/layer for: (a) today's 7-launch graph;
  (b) one persistent 64-WG launch that calls the existing device bodies with **no inter-stage waits**
  (wrong results, the same ablation trick the repo already uses, `qwen4exp.md:293-296`) -- this is the
  floor any megakernel/MoE-persistent design can reach; (c) the same with real counter edges.
  **Expected and decision rule:** if (b) is not >=15% below (a) (>=0.45 ms/token, ~3% tok/s, which
  would extrapolate to ~1.5 ms across GDN/HC/QSA layers [INFERENCE]), the bubble term is too small: stop, pursue
  section 4.1-4.3. If (b) passes but (c) gives back more than half, only per-task counters with
  early weight loads justify continuing. If (c) also holds >=15%, build the MoE-block persistent kernel
  (4.4) first, still not the whole model.
- Keep every experiment on device 1 (headless) and under `tools/tp/tdr_watch.psm1`-style TDR checks.

## Sources

[1] Spector et al., "Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B", Hazy Research, 2025-05-27 -- https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles ; code https://github.com/HazyResearch/Megakernels
[2] Cheng et al., "MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs" (OSDI'26), arXiv:2512.22219 -- https://arxiv.org/abs/2512.22219 (HTML https://arxiv.org/html/2512.22219); code https://github.com/mirage-project/mirage
[3] Chowdhary et al. (AMD Research), "Fleet: Hierarchical Task-based Abstraction for Megakernels on Multi-Die GPUs", arXiv:2604.15379 -- https://arxiv.org/abs/2604.15379 (HTML https://arxiv.org/html/2604.15379v1)
[4] Li et al., "ForgeMegakernel: A General Framework for Efficient Auto-Regressive Model Decode Megakernels", arXiv:2609.12379 -- https://arxiv.org/abs/2609.12379 (HTML https://arxiv.org/html/2609.12379)
[5] Arledge, MegaQwen (README, 3090 numbers, grid.sync cost) -- https://github.com/Infatoshi/MegaQwen
[6] AlpinDale, qwen_megakernel README -- https://github.com/AlpinDale/qwen_megakernel
[7] AlpinDale, "Hitting 1,000 tokens per second on a single RTX 5090" (2026-02-09) -- https://blog.alpindale.net/posts/5090_decode_optimization/
[8] ROCm HIP docs, "Cooperative groups" -- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/cooperative_groups.html
[9] ROCm-Device-Libs OCKL `cg.cl` (`__ockl_grid_sync`, `AVOID_GWS`) -- https://raw.githubusercontent.com/ROCm/llvm-project/amd-staging/amd/device-libs/ockl/src/cg.cl
[10] HIP SDK for Windows (documented as a subset of ROCm) -- https://rocm.docs.amd.com/projects/install-on-windows/en/develop
[11] LLVM "User Guide for AMDGPU Backend" (gfx12 memory-model scope table, `llvm.prefetch`) -- https://llvm.org/docs/AMDGPUUsage.html
Also read (kernel-level only, not megakernels): HipKittens, arXiv:2511.08083 -- https://arxiv.org/abs/2511.08083

Repo references: `docs/qwen4exp.md` (Decode performance, Decode kernel inventory and fusions),
`docs/r9700.md`, `docs/research/r9700-arch.md`, `docs/research/r9700-kernel-lessons.md`,
`docs/trellis-kernel.md`, `docs/tp.md`. Local SDK: `C:\opt\rocm\include\hip\{hip_runtime_api.h,
amd_detail\hip_cooperative_groups_helper.h, amd_detail\device_library_decls.h}`.
