# tools/tp_bench -- TP=2 all-reduce feasibility benchmark

`ar_bench.exe` decides whether tensor parallel TP=2 across the box's two R9700s (gfx1201, no peer
access, isLargeBar 0) is worth building. The only fast all-reduce without P2P is a kernel-driven,
zero-copy, one-shot PUSH through pinned host memory that both GPUs map. This tool measures that
path hop by hop, and then inside an emulated decode step. It is a hardened fork of libr4d's
`third_party/libr4d/r4d_ar_oneshot_2rank_exact.hip`. The top-of-file comment in `ar_bench.hip` has
the protocol, the memory-ordering argument for each variant and the safety properties.

Targets from the TP=2 plan: 1.5x needs the in-engine latency `L(10 KiB) <= ~7-12 us`, DFlash2
verify needs `L(80 KiB) <= ~15 us`, and break-even is about 80 us. The 640 KiB rows are the
prefill all-reduce.

## Measured on this box (2026-09-24, both R9700s, HIP_VISIBLE_DEVICES unset, desktop live on dev0)

| stage | result |
|---|---|
| probe | `hipHostMalloc(Coherent\|Mapped\|Portable)` maps into both devices at the host address; peer access 0/0; `ComputePreemptionSupported=0` on both |
| loopback RTT (p50 / p99) | dev1 3.28 / 4.40 us, dev0 2.80 / 3.52 us, dev1 with a CPU partner 2.00 / 2.40 us |
| pingpong dev0 <-> dev1 | one-way p50 **1.24 us** (RTT p50 2.48, p99 3.40 us) |
| isolated 10 KiB, us/AR | flag(d1,a1) 6.40 at nb=1, 10.4 at nb=8; **flag(d3,a0) 5.34 at nb=4**; ll 29-30 |
| isolated 80 / 640 KiB, us/AR | flag(d3,a0) 11.7 / 58.3 at nb=4; ll 199 / 1758+ (ll is out) |
| decode emulation, L per AR | **flag(d3,a0) nb=4: 10 KiB 7.47 +- 0.38 us, 80 KiB 13.33 +- 0.38 us** (vs no AR kernel); flag(d1,a1): 10.27 / 16.88; ll: 38.0 / 192.7 |
| stress, flag(d3,a0) nb=4, all bit-exact | 10,000,000 isolated at 10 KiB (~146k AR/s), 1,280,000 under the decode pattern, 1,000,000 at 80 KiB |
| stress, other variants, all bit-exact | 1M each of flag(d1,a1), flag(d3,a0) nb=8, ll; 128k flag(d1,a1) decode |

Decision: flag(d3,a0), nb=4 passes every gate (L(10 KiB) <= 12 us, L(80 KiB) <= 15 us). With the
measured per-rank GEMM sum (`tune_gemm.py`'s `sweep_shape`: 12.7-13.1 ms per rank vs 23.0-23.4 ms
single-GPU at M=1/8), it projects plain decode at 18.0-18.6 ms/token, 1.50-1.55x the single-GPU
27.8 ms.

## Build (CPU only; never runs the binary)

    powershell -File tools\tp_bench\build.ps1          # -> tools\tp_bench\build\ar_bench.exe
    powershell -File tools\tp_bench\build.ps1 -Asm     # also build\ar_bench-gfx1201.s (device ISA)

`build/` is in-tree, like `tools/bench/build/`. The repo-wide `build/` rule in `.gitignore`
already ignores it.

## Modes (run in this order; each one runs on its own)

| mode | kernels? | measures |
|---|---|---|
| `probe` | none | Per device: name, PCI id and the host-mapping, atomic, preemption, large-BAR and wall-clock attributes. Peer access in both directions. A 4 MiB shared mailbox through path A (`hipHostMalloc(Coherent\|Mapped\|Portable)`) and path B (`VirtualAlloc` + `hipHostRegister(Mapped\|Portable)`): does `hipHostGetDevicePointer` work on each device, and does the device pointer equal the host pointer. Exits 1 if the path selected by `--alloc` is not usable on both devices. |
| `loopback --device D [--partner gpu\|cpu]` | 1 device | Round trip GPU -> host memory -> GPU on one card. With `gpu`, 2 blocks of one kernel ping-pong a seq value. With `cpu`, a CPU thread echoes it. RTT in ns from in-kernel `wall_clock64`. |
| `pingpong` | both | dev0 <-> dev1 through the shared host mailbox, one single-block kernel per device. RTT and one-way (= RTT/2), no launch overhead. |
| `allreduce --pattern isolated` | both | Back-to-back all-reduces. Mean us per AR from hipEvents (max over ranks). `--trace` adds block 0's in-kernel push/wait/reduce/total ns (and a block-0 barrier, so take headline numbers without it). |
| `allreduce --pattern decode` | both | Per token: 64 layers x [filler, slot, filler, slot]. The filler streams `--filler-mb` MiB from a 512 MiB hash-filled VRAM ring and read-modify-writes `--dirty-mb` MiB (one wave of resident blocks, so those lines are dirty at AR time). Three conditions run **interleaved token by token** in one timed plan: (a) the real AR in the slot, (b) a local stand-in kernel on the same grid, (c) fillers only. The input-generating kernel runs before every token of every condition. **`L = (T_a - T_b) / 128`** us per AR (the spec's number) and the conservative `(T_a - T_c) / 128`, each with a standard error from the paired per-token differences. |
| `stress --count N` | both | N verified all-reduces. Stops at the first mismatch, timeout or protocol violation and reports call, block, word, expected and got. Prints progress every ~5 s. `--pattern decode` runs the same check under decode conditions (filler traffic and dirty L2 between ARs): the count is in ARs, rounded up to whole 128-AR tokens. |
| `sweep` | both | probe; loopback on dev1 then dev0; pingpong; then for each variant, safe one first (flag(d1,a1), flag(d3,a0), ll): isolated at {10240, 81920, 655360} B x nb {1,4,8,16,32} (2000 iters), then right away its decode rows at 10 KiB and 80 KiB with the best isolated nb. The safe variant's decode L is in the JSON before a relaxed variant can end the sweep. Final tables plus a decision block. |

Knobs: `--variant flag|ll`, `--drain 1|3`, `--acq 0|1`, `--spin rlx|acq` (flag polls: relaxed
polls plus one acquire fence, the default, or an acquire load on every poll), `--bytes N`
(multiple of 16, <= 1 MiB), `--nb`, `--nt`, `--iters`, `--graph` (per-rank hipGraph,
ThreadLocal capture; captured, instantiated and uploaded before the start barrier),
`--timeout-ms` (default 100, hard cap 500), `--tokens`, `--filler-mb`, `--dirty-mb`,
`--alloc hostmalloc|register`, `--dev0/--dev1`, `--rounds-per-launch` (0 = calibrate), `--json <path>`.

**drain 3 note.** The build runs in WGP mode. There, `__syncthreads`' workgroup release already
emits `s_wait_storecnt 0` in every wave, so `--drain 3` produces the same ISA as no drain at all.
The flag(d3,a0) rows therefore measure "no drain, tid0's release writeback only". They are correct
because tid0's system-scope release starts with `global_wb scope:SCOPE_SYS`, which is L2-wide and
writes back the other waves' pushed lines. Under CU mode (`-mcumode`) the barrier lowering would
differ.

Exit codes: `0` ok, `1` error (including a host-barrier timeout, a run that stopped early with no
recorded cause, or a WallClockRate that disagrees with the host clock), `2` data mismatch or
protocol violation, `3` timeout / abort / stuck stream / ping launch budget exceeded.

## Exact commands (staged, HIP_VISIBLE_DEVICES unset)

    tools\tp_bench\build\ar_bench.exe probe --json probe.json
    tools\tp_bench\build\ar_bench.exe loopback --device 1 --json lb1.json
    tools\tp_bench\build\ar_bench.exe loopback --device 1 --partner cpu --json lb1cpu.json
    tools\tp_bench\build\ar_bench.exe loopback --device 0 --json lb0.json
    tools\tp_bench\build\ar_bench.exe pingpong --json pp.json
    tools\tp_bench\build\ar_bench.exe allreduce --iters 5000 --json ar_flag.json
    tools\tp_bench\build\ar_bench.exe allreduce --iters 2000 --trace --json ar_flag_trace.json
    tools\tp_bench\build\ar_bench.exe allreduce --variant ll --json ar_ll.json
    tools\tp_bench\build\ar_bench.exe stress --count 1000000 --json stress_flag.json
    tools\tp_bench\build\ar_bench.exe sweep --json sweep.json
    tools\tp_bench\build\ar_bench.exe stress --drain 3 --acq 0 --count 10000000 --json stress_d3a0.json
    tools\tp_bench\build\ar_bench.exe stress --variant ll --count 10000000 --json stress_ll.json
    tools\tp_bench\build\ar_bench.exe stress <winning variant> --nb <its nb> --pattern decode --count 1280000 --json stress_dec.json

If path A fails on dev1 in `probe`, add `--alloc register` to every later command. The sweep's
`summary.decision.requires_stress_validation` prints the exact decode-stress command for the
variant that won the 10 KiB decode L.

## Safety properties

- Every spin loop uses `s_sleep`, a wrap-safe compare `(int)(cur - want) >= 0`, and a
  `wall_clock64` timeout of 500 ms or less, well under the 2 s Windows TDR. Every 16 polls it also
  reads both ranks' ABORT words and returns at once if either is set.
- On timeout a block release-stores its rank's ABORT word, records block, seq and phase, and
  returns without reducing. It also sets the rank's VRAM **sticky** word. Every later AR kernel of
  that rank reads it at entry and skips the whole call (no push, no flag, no wait, no reduce).
  An aborted rank therefore never pushes again, so its in-flight batches cannot overwrite a slot
  the stalled peer has yet to read. A flag or LL seq that is 2+ calls ahead is checked against
  the abort words: it is either a bail-out or a genuine PROTOCOL_VIOLATION (exit 2).
- The host checks both ABORT words and the per-rank Status after every batch. Nothing continues
  after an abort. Mismatches at or after the first failed call are labelled
  `post_abort_consequential`.
- Ping kernels (loopback/pingpong): a 100-round calibration launch, then about 40 ms per launch
  (16..20000 rounds), and an **in-kernel budget**. The initiator starts no new round 250 ms after
  kernel entry (the responder: 300 ms) and fails with `ping-launch-budget-exceeded` instead. A
  launch therefore lasts at most budget + one timeout, under 1 s.
- `hipDeviceAttributeWallClockRate` is checked against the host clock once per device, before the
  first spinning kernel. A busy kernel, bounded by an iteration count, reads the counter at entry
  and exit. More than 5% off gives exit 1: every timeout and ns figure depends on this rate.
- Memory layout: every flag, ABORT word and per-block mailbox region starts on its own 128 B line.
  A rank's mailbox (data and flags) is written only by the peer. ABORT and Status-mirror lines are
  written only by their owner. No line is ever written by both devices.
- seq counters are per-block u32 values in VRAM, incremented by the kernel, so hipGraph replay is
  safe. Slots are double-buffered by seq parity. Before every session the shared region is zeroed
  while both GPUs are idle. Each session also starts its seq counters above every seq an earlier
  session in the process used, so a stale flag or LL half is always "behind" even if a zeroing
  memset was not snooped.
- No hipMalloc, hipFree or device sync inside timed loops. Every host wait polls
  hipEventQuery/hipStreamQuery under a 30 s watchdog, and a stuck stream is reported (exit 3), not
  waited on forever. Host threads never wait on the peer's stream. They only meet at host barriers,
  which time out, and a barrier timeout is an error (exit 1), never a silent early "ok". Each rank
  enqueues its next batch before any host-side printing, so a stalled console idles both GPUs
  instead of leaving one rank's batch unpaired.
- hipMemcpyPeerAsync is never used.

## Correctness data

Rank r's input for call c, element j is `(hash(c, j, r) % 97) - 48` in bf16. The hash is a 32-bit
mixer of the full call index, the element and the rank. Every sum is exact in bf16, so
verification is bit-exact. Stale data (by any number of calls) or a wrong block offset matches only
by chance, about 1/97 per element.

Verify cadence (a deliberate deviation from the spec's "after every call" / "every <= 64 calls"):
every output of every AR is verified, once per batch. Isolated and stress use batches of 50 ARs;
decode uses one 128-AR token. Every call has its own ring row, so nothing goes unverified. It is
also the stronger stress test, since no verify kernel sits between ARs.

The input-generating and verify kernels run **outside** the hipEvent pair of each batch, so their
cost is excluded, not subtracted. Isolated timing discards 4 leading batches (200 ARs), after one
throwaway AR that follows the first host barrier. Decode discards one token per condition.
Isolated `--iters` is rounded up to a multiple of 50.

## Reading the JSON

- Top level: `mode`, `argv`, `exit_code`, `wallclock_check[]` (reported vs measured kHz per
  device), plus one object per mode (`loopback`, `pingpong`, `allreduce`, `stress`) or, for
  `sweep`, `probe` / `loopback[]` / `pingpong` / `isolated[]` / `decode[]` / `summary`.
- loopback / pingpong: `rtt_ns` {n, mean, stdev, min, p50, p90, p99, p99_9, max}, `one_way_ns`
  (pingpong), `rtt_ns_samples` (every sample; the calibration launch and the first round of every
  launch are excluded), `launches`, `rounds_per_launch`, `launch_budget_ms`, `spin`. On failure,
  `failure` (loopback) or `ranks[].status` (pingpong) with the code (`timeout`,
  `ping-launch-budget-exceeded`) and, for loopback, `responder_block_started`.
- allreduce / stress: `config`, `status` (`ok|mismatch|protocol_violation|timeout|error`),
  `protocol_violation`, `first_failed_call_any_rank`, `ranks[]`, `result`.
  - Isolated: `result.mean_us_per_ar_max_over_ranks` is the headline number. Per rank:
    `mean_us_per_ar`, `batch_ms[]` (50 ARs each), `batch_us_per_ar` stats,
    `host_starved_batches` (batches enqueued after the GPU had already gone idle; should be 0),
    and `trace` when `--trace` is set. In the trace, for flag `push` runs from kernel entry to
    after the flag release store (push, drain, barrier and the release's system writeback) and
    `wait` is the spin. For LL, `wait` includes the fused reduce.
  - Decode: `result.L_us_in_engine` +- `L_us_in_engine_se` is **the** number (vs the stand-in).
    `result.L_us_vs_no_ar_kernel` +- se is the conservative one (it also charges the AR slot's own
    dispatch). `per_token_ms_{allreduce,standin,fillers_only}_max_over_ranks` give T_a, T_b, T_c.
    Per rank: the same numbers, `L_us_by_quintile` (drift check), token time stats, raw
    `batch_ms_*`, and `filler_grid`.
  - Stress: `result.calls_verified`, `calls_requested_rounded`, `seconds`, `calls_per_s`.
  - Failures: `ranks[].failure` holds `abort_words`, `first_failure` {code, protocol_violation,
    block, seq, call_index, call_index_in_timed_run, phase, word, observed}, `n_abort_exits`,
    `n_skipped_blocks`, `first_failed_or_skipped_call_this_rank`, `verify_mismatches` and
    `first_mismatch` {call_index, element, block, payload_word, wire_word_in_block (LL),
    expected, got, post_abort_consequential}.
- sweep `summary`: `isolated_table` (us/AR per nb, null = failed), `decode_table` (nb, T_a, T_b,
  T_c, both L values with se), and `decision`. The decision block holds the best L at 10 KiB and
  80 KiB (both definitions) against the 7 / 12 / 15 / 80 us thresholds, `projected_tok_ms_*` and
  `projected_speedup_*` (27.8 / (17.0..17.6 + 0.128 x L)), `requires_stress_validation`, and
  `baseline_note`. `baseline_note` says which L matches how the 17.0-17.6 ms zero-comm step was
  measured.
