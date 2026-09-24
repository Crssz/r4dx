# TP=2 tensor parallel for r4dx -- implementation design

Branch `tp2` (worktree `C:\Users\pay20\dev\r4dx-tp2`, from `main` at `aa54c20`). Written 2026-09-24.
This is the design coding agents follow **literally**. Where it says MUST, do exactly that; where it
names a file, function or signature, use that name. Anything not covered here is a question for the
user, not a judgment call.

Inputs this design rests on (all read before writing it): `README.md`, `docs/architecture.md`,
`docs/container-format.md`, `docs/sampling.md`, `docs/mtp.md`, `docs/dflash2.md`, `docs/vision.md`,
`docs/server.md`, every file in `src/model`, `src/core`, `src/kernels`, `src/cli`, `src/server`,
`tools/tp_bench/README.md` and `tools/tp_bench/ar_bench.hip`, plus the research JSONs
(`map0..3.json`, `synthesis.json`, `challenges.json`). Line numbers below refer to `aa54c20`.

**Revision 2** (same day) applies two design reviews. Every blocker and major finding was checked
against the code; the accepted ones are folded into the sections below, and the rejected parts are
in Appendix B with the evidence. The largest changes: the server's error recovery (2.4, 8.4 -- one
reviewer blocker plus a pre-existing `PrefixState::Invalidate` bug on `main` that the reviewers did
not catch), the exact condvar hand-off and a progress-based watchdog (2.2), the phase order
(P2 split into P2a/P2b with P3 between them, section 11), the TP-only tuning table (2.7), the
two-GPU test opt-in (10.1) and DFlash losslessness gates under TP (G12).

---

## 0. Decisions at a glance

| Question | Decision | Why (short) |
|---|---|---|
| Parallelism shape | Megatron TP=2: column-parallel in-projections, row-parallel out-projections, one all-reduce after each row-parallel output at 3 code sites (GDN `out_proj`, attention `o`, MLP `down`) = 2 per layer, 128 per decode token | kernels already accept every per-rank shape (gqa 6, GDN H/Hg 3, N%16, K%(SK*group)) |
| Process model | one process, one **rank thread per GPU** (`hipSetDevice` first, owns its own `Model`), one facade thread (CLI main / server worker) that broadcasts every call to both rank threads and waits | no src file calls `hipSetDevice`; all GPU state is per `Model`; HIP's current device is per thread |
| How TP reaches the model code | `Model` becomes **SPMD-aware**: it runs on a rank-local config, takes a `core::TpComm*`, and does every cross-rank merge itself. The facade (`TpModel`) never computes; it broadcasts, reconciles and error-handles | every Model code path (plain, sampled, MTP, DFlash, verify) works under TP with no second implementation |
| All-reduce transport | port of `tools/tp_bench`'s `flag(drain3, acq0)` kernel, `nb=4`, `nt=256`, one `hipHostMalloc(Coherent|Mapped|Portable)` mailbox, two fixed-geometry channels (<=17 rows / <=64 rows) | measured +7.47 us per 10 KiB, +13.3 us per 80 KiB in a decode emulation, 12.28M bit-exact |
| lm_head | **vocab-split**, 124160 rows per rank | saves ~0.5 ms/token plain and ~1.1 ms per DFlash round (verify head + drafter head), -0.33 GiB per rank; the merges are exact and cheap (8 B/row greedy, 536 B/row sampled) |
| Greedy merge | each rank returns `(max value, lowest local index)`; host all-gather; larger value wins, tie -> rank 0 (lower global id) | identical to full-row argmax with lowest-index tie-break |
| Sampled merge | per-shard device row summary (top-64 + lse); host all-gather; merged summary = top-64 of the 128 under canonical order, `lse = logaddexp`; unresolved rows gather the full row | exact top-64; lse error stays inside the existing 1e-3 band, so sampling stays lossless |
| MTP | MTP head **sharded like a body layer** (same hooks), full-vocab draft argmax merged per draft step on the host; reduced-vocab draft head (if used) replicated | reuses the body sharding and hooks; ~0.1-0.2 ms/round of host hops |
| DFlash2 | drafter **replicated on both ranks**; target lm_head (split) over the 8 draft rows, per-row top-16 merged on the host before the selector walk | the drafter is launch-bound (3.26 ms of 5.77); sharding it buys ~1 ms and costs ~12 extra all-reduces and a second sharding scheme |
| Vision | tower on **rank 0 only**; merged rows are copied to a host buffer and spliced by both ranks with an H2D copy | 0.9 GiB once; no device pointer crosses ranks |
| Graphs | **no hipGraph in v1** | branch `m12-hipgraph` measured +0.4% |
| Emulation | both shards on one device, two rank threads, host-synchronous exact add at every all-reduce | the byte-exact reference for real 2-GPU runs and the numerics check vs TP=1 on device 1 alone |
| TP=1 | every TP branch is behind `comm_ != nullptr` / `tp_world > 1`; `--tp` absent => today's bytes | gate from P2a on (`tools/tp/tp1_identity.ps1`) |
| TP tunings | new `(N,K)` rows go in a **separate** `gemm_tuning_table_tp2.inc`, consulted only on threads that loaded a TP rank | one TP=2 key (`17408 x 5120` w4a16) is also the DFlash drafter's `gate_proj`/`up_proj`; a shared table would change TP=1 DFlash bytes |
| Error recovery | every error -> `kNeedsRecovery`; `Reset()` heals; host-only policy calls (`SetDflashInjectionEnabled`) and cached accessors stay legal in that state; the server's `PrefixState::Invalidate` is fixed so the next request really calls `Reset()` | the production `--dflash` server must recover from any TP error without a restart (2.4, 8.4) |
| Phase order | P1 -> **P2a** (loader, noop comm, per-rank step: G3) -> **P3** (all-reduce: G5) -> **P2b** (emulation, `TpModel`: G4) -> P4 -> P5 | both economic gates are known before the ~2,700 LOC of threading/emulation/`TpModel` code (P2b) is written |

---

## 1. Goals, non-goals, success gates

### 1.1 Goals

1. Single-stream decode speedup on the production config (v6 container, `--layout w4a16`, group 64,
   attn.k/v bf16, 4-bit lm_head) using both R9700s: **>= 1.45x plain greedy** and **>= 95 tok/s
   with `--dflash ... --dflash-k 7`**.
2. Feature parity under `--tp 2` by the end of P5: plain greedy, plain sampled (device row
   summary), chunked prefill, `--chat` prefix reuse, `Model::Reset`, MTP (greedy + sampled), DFlash2
   (greedy + sampled), vision (`--image`, server `image_url`), the OpenAI server including tool calls.
3. **Lossless sampling** under TP: the emitted token for a given set of logits and draw `u` is exactly
   `SampleCanonical(full row, u)` (docs/sampling.md section 2); speculative rounds keep one draw per
   emitted token.
4. **Real 2-GPU TP is byte-identical to single-device emulation** of TP=2 (same kernels, same
   tunings, exact fp32 all-reduce).
5. **`--tp` not given => main's behaviour byte for byte** (text, token ids, teacher-forced dumps).
6. Failure containment: an all-reduce never spins longer than 500 ms (TDR is 2 s); any rank error
   surfaces as a C++ exception on the facade thread; `Reset()` recovers the group.

### 1.2 Non-goals (v1)

- TP > 2, pipeline parallel, data parallel, batching / multi-sequence.
- P2P, IPC, RCCL, `hipMemcpyPeerAsync` (corrupts on this box) -- never used.
- hipGraph capture of the decode step (measured +0.4%). If ever added: `hipStreamCaptureModeThreadLocal`.
- Fusing all-reduce into `r4dx_residual_rmsnorm_bf16` (it is one workgroup per row at T=1; a similar
  collapse measured -4.3%, `linear.cpp:121-129`). Separate kernels in v1.
- Bit-identity with TP=1 (impossible: row-parallel partials are rounded to bf16 before the sum, and
  split-K order changes). TP=1 comparison is KL-based.
- A re-converted per-rank container. Shards are gathered from the one `.r4dx` file at load time.
- Sharding the DFlash2 drafter or the vision tower.
- `--profile` / `--profile-prefill` under TP (rejected at arg parse).
- Linux.

### 1.3 Established facts used as inputs (measured 2026-09-24, trusted)

| Fact | Value |
|---|---|
| Production today | 35.98 tok/s plain, 73.6 tok/s `--dflash k=7` (24.9% acceptance, 2.71 tok/round); standard protocol below |
| Per-rank GEMM sum at TP=2 shapes (tuned sweep) | 12.7-13.1 ms/token vs 23.0-23.4 ms at TP=1 (M=1 and 8) |
| All-reduce, `flag(d3,a0) nb=4`, decode emulation | L(10 KiB) = 7.47 +- 0.38 us, L(80 KiB) = 13.33 +- 0.38 us -- these are tp_bench's **`L_vs_no_ar_kernel`** (conservative) values (`sweep.json` `summary.decision.best_L_vs_no_ar_kernel_us_{10,80}KiB`); the vs-stand-in values are 8.26 / 11.94 us |
| All-reduce, isolated 640 KiB | 58.3 us at nb=4 |
| Stress | 12.28M all-reduces bit-exact (10M isolated 10 KiB, 1.28M decode-pattern, 1M at 80 KiB) |
| Devices | no P2P; HIP device 0 drives the desktop (`ComputePreemptionSupported=0`), device 1 headless; pinned `Coherent|Mapped|Portable` memory maps into both devices **at the host address** |
| hipGraph | +0.4% (branch `m12-hipgraph`) -- not worth it |

**Standard protocol** (docs/perf.md, used by every speed gate here): prompt
`"Write a haiku about GPUs, then explain what a GPU is in two sentences."`, `--layout w4a16 --vision off
--think off --temperature 0 --max-tokens 256 --max-ctx 2048 --stats`, run twice, both numbers reported.

### 1.4 Performance model (what the gates are calibrated on)

Per-rank decode step with a free all-reduce = per-rank GEMM sum (12.7-13.1 ms) + non-GEMM work
(~4.0-4.5 ms: the elementwise work over `hidden` is replicated, and several GDN kernels are occupancy-
bound at half the heads -- `recurrent_update` fused-norm runs 24 workgroups per rank) = **16.7-17.6 ms**.

**Which L goes with which baseline.** v1 keeps the all-reduce as a separate, extra kernel (1.2), and
the P2a step measurement (G3) uses `NoopComm`, which launches **no** kernel at the all-reduce sites.
tp_bench's condition (c) "fillers only, no AR-slot kernel" is exactly that baseline, so the matching
latency is `L_vs_no_ar_kernel = (T_a - T_c) / 128` -- the 7.47 / 13.33 us used below. G5 gates the
engine's port on the same definition (and reports the vs-stand-in value next to it).

| Scenario | TP=1 today | TP=2 projection | Gate |
|---|---|---|---|
| plain greedy, 2k | 27.8 ms (35.98 tok/s) | 16.7-17.6 + 128 x 7.47 us (0.96 ms) + ~0.05 ms host merge = 17.7-18.6 ms -> **53.8-56.5 tok/s (1.50-1.57x)** | >= 52.2 tok/s (1.45x), P4 |
| `--dflash k=7` | 36.8 ms/round (73.6 tok/s) | draft 6.3 - 0.55 (half lm_head) + verify ~17.7 + 128 x 13.3 us (1.7 ms) + ~0.4 inject/merge = ~25.5 ms -> **~105 tok/s** | >= 95 tok/s, P5 |
| `--mtp 3` | 36.4 ms/round (65.93 tok/s) | ~21-23 ms/round -> ~100-110 tok/s | reported, not gated |
| prefill, short prompt | ~1000 tok/s | per 64-token chunk: ~39-41 ms compute + 128 x 58 us (7.4 ms) -> ~1330-1390 tok/s (1.33-1.39x) | reported |
| prefill, 131k ctx | 423.5 tok/s | prefill attention grid is `(1, kv_heads_local=2)`: does not speed up -> ~1.15x | reported |
| plain decode, 131k / 262k | ~29.6 / ~24.9 tok/s | KV read halves per rank -> ~1.6 / ~1.7x | reported |

### 1.5 Success gates (numbers)

| # | Gate | Threshold | Phase |
|---|---|---|---|
| G1 | CPU tests | `slice(pack(W)) == pack(slice(W))` for every layout/axis/fused tensor; mxfp4 K-slice `wref == full wref`; vocab merges; host exchange; rank-worker hand-off (1M commands through the condvar) | P1 |
| G2 | TP=1 byte identity | every row of the `tp1_identity.ps1` matrix SHA-256-equal to the frozen baseline | P2a (re-run P3, P2b, P4, P5) |
| G3 | Per-rank step, no-op all-reduce, device 1 | median(rank r) <= **0.633 x** median(TP=1) for r = 0 AND 1, both measured by `tool_tp_step_bench` in the same session on the same device (v6, w4a16, greedy, `--max-ctx 2048`). 0.633 = 17.6 / 27.8. The absolute ms are recorded, not gated | P2a |
| G4 | Emulated TP=2 numerics | mean KL(ref ‖ TP2emu) <= **0.0435** (TP=1: 0.03851) and top-1 >= **90.43%** (TP=1: 90.93%) on `tools/reference/kl_corpus` | P2b |
| G5 | In-engine all-reduce | 0 mismatches in 10M stress ARs; in the decode pattern, `L_vs_no_ar_kernel`(10 KiB) <= **9.0 us** and (80 KiB) <= **16.0 us** (<= 120% of tp_bench's 7.47 / 13.33) | P3 |
| G6 | Real 2-GPU == emulation | every `*.logprobs.f16` of the teacher-forced corpus SHA-256-equal, both produced by the **same binary** in the same gate run | P4 |
| G7 | Plain decode speed | >= **52.2 tok/s** on the standard protocol, both runs | P4 |
| G8 | Soak | 60 min `tool_tp_soak`, exit 0, 0 aborts, 0 divergences, 0 TDRs (WER LiveKernelEvent 141 in the Application log, or System event 4101; N44) | P4 |
| G9 | DFlash speed | >= **95 tok/s** `--dflash k=7`, standard protocol, both runs | P5 |
| G10 | DFlash acceptance | mean tokens/round over the 4 prompts of `tests/model/mtp_prompts.txt` within **±5% relative** of TP=1 | P5 |
| G11 | Server | `tools/server/smoke.ps1 -Tp 2` clean (4-layer default; real container with `-Dflash -ToolRoundTrip`; `-Vision -Dflash`; `-Mtp 3`; **`-Dflash -TpFault`**: an injected all-reduce fault fails one request and the next request succeeds with the reference output) | P5 |
| G12 | Speculative losslessness under TP | `tools/validate_dflash.ps1 -Tp 2 -Layouts w4a16` and `tools/validate_spec_sampling.ps1 -Tp 2 -Layouts w4a16` exit 0 under the scripts' own control rules; `test_tp_emulation`'s DFlash and MTP cases (exact H6/H7 merges, bookkeeping lockstep) pass | P5 |

---

## 2. Process and threading model

### 2.1 Shape

```
facade thread (r4dx-cli main thread / r4dx-server Engine worker thread)
  |  TpModel::DecodeStepGreedy(tok)  -> RunAll(cmd)  -> wait -> reconcile -> return
  |
  +-- rank thread 0  (hipSetDevice(devices[0]) first)   owns Model[rank 0], TpComm endpoint 0, streams
  +-- rank thread 1  (hipSetDevice(devices[1]) first)   owns Model[rank 1], TpComm endpoint 1, streams
                        |                                     |
                        +---- all-reduce kernels <-- pinned host mailbox --> all-reduce kernels
                        +---- HostExchange (in-process rendezvous, CPU memory) ----+
```

- The facade thread **never** calls HIP after load (no `hipSetDevice`, no allocation, no
  `hipMemGetInfo`). Everything device-related runs on a rank thread.
- Rank threads are persistent for the life of the `TpModel`. A rank's `Model`, its `TpComm`
  endpoint and every `DeviceBuffer` it owns are **constructed and destroyed on that rank's thread**.
- There is exactly one facade thread issuing calls, exactly as `Model` today is single-worker
  (`model.h:6-9`). `TpModel` is not thread-safe except for the cached const accessors listed in 2.8
  (`HasVision()` is read by HTTP threads, `engine.h:115`).

### 2.2 Rank thread (`class RankWorker`, `src/model/tp/tp_rank_worker.h`, header-only, no HIP)

`RankWorker` knows nothing about HIP or `Model`, so the hand-off is CPU-testable in P1
(`tests/model/test_tp_rank_worker.cpp`). `TpModel` owns one `RankWorker` per rank plus that rank's
`Model` and `TpComm` (in a `RankSlot` struct next to it).

```cpp
namespace r4dx::model::tp {
struct WorkerTiming {
  std::chrono::microseconds rank_spin{20000};    // rank spins this long before blocking
  std::chrono::microseconds facade_spin{200};    // facade spins this long before blocking
};
class CompletionGroup {                          // shared by the facade and every worker
 public:
  std::mutex mu; std::condition_variable cv;     // the facade blocks here
};
class RankWorker {
 public:
  // Starts the thread. Its first action is `thread_init()` (TpModel passes
  // [dev]{ R4DX_HIP_CHECK(hipSetDevice(dev)); SetThreadDescription(L"r4dx-tp-rank<r>"); }).
  RankWorker(int rank, std::function<void()> thread_init, CompletionGroup* done_group,
             WorkerTiming timing = {});
  ~RankWorker();                                  // see 2.6; never posts to a busy worker
  void Post(std::function<void()> cmd);           // facade only; precondition Idle()
  bool Idle() const;                              // done_ == posted_
  uint64_t Posted() const, Done() const;
  std::exception_ptr TakeError();                 // exception of the last cmd, or null
  std::atomic<uint64_t>& Heartbeat();             // bumped by the rank's code, read by the facade
 private:
  void ThreadMain();
  std::thread th_;
  std::mutex mu_; std::condition_variable cv_;    // the rank blocks here
  std::atomic<uint64_t> posted_{0}, done_{0}, heartbeat_{0};
  std::function<void()> cmd_; std::exception_ptr err_; bool exit_ = false;
  CompletionGroup* group_; WorkerTiming timing_;
};
}
```

**Command protocol (lockstep at the command level).** C++17: no `std::atomic::wait`, so every
blocking hand-off publishes under the mutex its waiter checks the predicate under. That rules out
the lost wakeup where the waiter tests the predicate, the publisher stores and notifies, and only
then does the waiter block.

1. **Post** (facade): `{ std::lock_guard<std::mutex> lk(mu_); assert(done_ == posted_); cmd_ =
   std::move(cmd); posted_.store(posted_ + 1, std::memory_order_release); } cv_.notify_one();`
2. **Rank wait**: `next = done_ + 1`. Spin while `posted_.load(acquire) < next`, for at most
   `timing_.rank_spin` (`_mm_pause`; `std::this_thread::yield()` every 64 spins). If still not
   posted: `std::unique_lock<std::mutex> lk(mu_); cv_.wait(lk, [&]{ return exit_ ||
   posted_.load(std::memory_order_acquire) >= next; });`. The spin keeps mid-generation hand-offs
   off the condvar (commands arrive every ~18 ms); an idle server blocks and burns no CPU.
3. **Run**: `try { cmd_(); comm->CheckHealthy(); } catch (...) { err_ = std::current_exception();
   if (!IsTpAborted(err_)) comm->Abort(kAbortHost, what); }` (the `CheckHealthy`/`Abort` calls are
   part of the closure `TpModel` builds; `RankWorker` only stores `err_`).
4. **Complete** (rank): `{ std::lock_guard<std::mutex> lk(group_->mu); done_.store(next,
   std::memory_order_release); } group_->cv.notify_all();`
5. **Facade wait** for command `seq`: spin up to `timing_.facade_spin` on `Done() >= seq` for every
   rank; then loop `{ std::unique_lock<std::mutex> lk(group->mu); group->cv.wait_for(lk, 1 s,
   all_done); }`, running the **progress watchdog** each time `wait_for` returns without all ranks
   done.
6. **Progress watchdog** (replaces a per-command timeout, which would kill a legitimate 100k-token
   prefill): the facade records `max over ranks of Heartbeat()` at every wake-up. Rank code bumps
   its heartbeat (relaxed `fetch_add`) at every `AllReduceSumBf16` enqueue and every `HostAllGather`
   / `Barrier` entry and exit -- which includes every `RunChunk` / `VerifyWindow` entry through the
   H1/H2 lockstep check. No other plumbing is needed: a 64-row prefill chunk takes well under 1 s
   even at 131k-262k context (~0.15 s on average at 131k), and a solo `EncodeImages` ~0.16 s per
   1024-token image -- both far below the 60 s threshold.
   If no rank's heartbeat has moved for **60 s**, the facade sets `state_ = kFatal`, logs every
   rank's `Posted/Done/Heartbeat`, and throws `TpTimeoutError`. It never posts again to a worker
   that is not `Idle()` (2.6). Every real wait inside a command is already bounded (500 ms
   all-reduce spin, 30 s `HostExchange`, 30 s `SyncWithWatchdog`), so the watchdog only fires on a
   thread stuck inside a HIP call.
7. Commands are either **collective** (`RunAll`: same closure on every rank; may use `TpComm`) or
   **solo** (`RunOne(rank)`: must NOT use `TpComm`; used for `EncodeImages` on rank 0 and for
   load-time helpers).

```cpp
// in TpModel. Fn: R(Model&, int rank). R may be void.
template <class Fn, class R = std::invoke_result_t<Fn, Model&, int>>
std::conditional_t<std::is_void_v<R>, void, std::vector<R>> RunAll(Fn&& fn);  // if constexpr on void
template <class Fn>
std::invoke_result_t<Fn, Model&, int> RunOne(int rank, Fn&& fn);
// Every forward entry point goes through RunCollective, which prepends the facade-cached host
// policy (2.4) to the closure on each rank: m.SetDflashInjectionEnabled(dflash_injection_).
template <class Fn> auto RunCollective(Fn&& fn);
```

### 2.3 Result reconciliation

After every collective command the facade:

- compares every rank's return value: `std::vector<int32_t>` / `int32_t` with `==`,
  `std::vector<float>` with `memcmp` (both ranks built it by the same all-gather, so it must be
  identical; ~50 us per 1 MB row), `walk_len` with `==`. Mismatch -> `TpDivergenceError`, state
  `kNeedsRecovery`;
- for methods that take `std::mt19937_64& rng`: before the command, copies the caller's generator into
  `rng_r` for each rank (`std::mt19937_64` is copyable); after, requires `rng_0 == rng_1`
  (`operator==` compares full state; mismatch -> `TpDivergenceError`) and assigns `rng = rng_0`;
- returns rank 0's value.

### 2.4 Error propagation and the state machine

`TpModel::state_` in {`kReady`, `kNeedsRecovery`, `kFatal`}.

| Event | What happens |
|---|---|
| A rank's closure throws anything | that rank calls `comm->Abort(kAbortHost, what)` (sets its host ABORT word in the mailbox and the `HostExchange` abort flag), so the peer's spinning all-reduce kernels exit within 16 polls and a peer blocked in `HostAllGather` wakes with `TpAbortedError` |
| Device all-reduce times out (500 ms) | kernel release-stores its rank's ABORT word, sets the VRAM sticky word; later ARs of that rank skip; the host sees the ABORT word at the next `CheckHealthy()` -> `TpAbortedError` with block/seq/phase from the VRAM `Status` |
| Host rendezvous timeout (30 s) | `TpTimeoutError`, group aborted |
| Results or rng differ | `TpDivergenceError` |
| Any of the above | facade picks the **root cause** = the first (lowest rank) exception that is not a `TpAbortedError`, else the first `TpAbortedError`; logs every rank's `what()` to stderr as `[r4dx-tp] rank <r> (dev <d>): <what>`; sets `state_ = kNeedsRecovery`; rethrows the root cause **with its original type** (the server's existing `catch` block handles it; with the 8.4 `PrefixState` fix the next request goes through `Reset()`) |
| A **device-work** call in `kNeedsRecovery` (the forward methods, `EncodeImages`, the profiled methods) | throws `TpStateError("tp: group aborted by an earlier error; call Reset() first")` |
| A **host-only** call in `kNeedsRecovery` or `kFatal` (table below) | runs normally; never throws a `TpStateError` |
| `Reset()` in `kNeedsRecovery` | runs **Recovery** (2.5), then `Model::Reset()` and the cached policy (`SetDflashInjectionEnabled(dflash_injection_)`) on every rank; `kReady` on success, `kFatal` + rethrow on failure |
| A device-work call or `Reset()` in `kFatal` | throws `TpStateError("tp: fatal, restart the process")` |

Every exception makes the group `kNeedsRecovery`, including deterministic precondition throws that
both ranks hit identically (e.g. `PagedKvCache::CheckCapacity` mid-forward). This is deliberate:
recovery costs < 1 s and removes every "was the comm still consistent?" judgment.

**Host-only calls** (legal in every state; they touch no device and no `TpComm`):

| Method | Implementation |
|---|---|
| `Config`, `ModelId`, `ImageTokenId`, `VisionMergeSize`, `HasVision`, `MtpEnabled`, `MtpUsingReducedVocabDraft`, `DflashEnabled`, `TpWorld`, `NumLoadedLayers` | values cached at load (2.9 step 10) |
| `PositionCount`, `SampledFallbackRows` | cached from rank 0 after every successful collective command |
| `SetDflashInjectionEnabled(b)` | stores `dflash_injection_ = b` on the facade only. `RunCollective` applies it on each rank at the start of every forward command, and `Reset()` applies it after recovery. `Model::SetDflashInjectionEnabled` is itself a host bool store (`model.h:590`), so applying it lazily is equivalent to applying it immediately |
| `Vram()` | `RunAll(hipMemGetInfo)`: no `TpComm`, and every rank is idle between commands, so it is safe in `kNeedsRecovery`; in `kFatal` it returns the last cached report |

Why the split matters: `Engine::RunRequest` calls `SetDflashInjectionEnabled` (`engine.cpp:367-369`)
**before** it decides between prefix reuse and `Reset()` (`engine.cpp:382-388`). If the toggle threw
in `kNeedsRecovery`, every request after the first TP error would throw before reaching `Reset()`,
and the production `--dflash` server would answer 500 until restarted. Section 8.4 has the second
half of that fix (`PrefixState`).

### 2.5 Recovery (`TpGroup::Recover()`, called from `TpModel::Reset()`)

1. `RunAll`: each rank `hipStreamSynchronize` on every stream it owns (Model's `stream_`, the vision
   stream on rank 0) using `SyncWithWatchdog(stream, 30 s)` (poll `hipStreamQuery`); a stuck stream
   -> `kFatal`. (Spinning kernels are bounded by the 500 ms timeout, so this terminates.)
2. `RunAll`: each rank D2H-copies its VRAM seq counters (both channels) and returns them.
3. Facade: `new_base = max(all seq values) + 64` (wrap-safe arithmetic in `uint32_t`).
4. `RunOne(0)`: zero the whole shared mailbox region (ABORT words, mirrors, flags, slots) with a host
   `memset` + `std::atomic_thread_fence(seq_cst)` -- both GPUs are idle. (The same zeroing also runs
   once right after the region is allocated, 2.9 step 7.)
5. `RunAll`: each rank `hipMemsetD32Async(seq_ch, new_base, 64)` for both channels, re-inits its VRAM
   `Status` (zero, then `abort_seq_min = 0xFFFFFFFF`), synchronizes.
6. **Host-side counters** (no rank inside any comm call): `HostExchange::Reset()` zeroes
   `arrive[r]` **and** the per-rank generation `gen_[r]` (both live in `HostExchange`, not in the
   endpoints, so one call resets them); `RunAll`: each endpoint's `ResetCounters()` zeroes its
   host call counters (`EmulatedComm`'s call index `c`, the fault-injection counter, `Stats()`
   deltas) and, for `EmulatedComm`, zero-fills `xchg[rank][0|1]`. Without this, an asymmetric abort
   (one rank threw before its counter increment, the other after) leaves the parities `c & 1` /
   `g & 1` different, and the next emulated add reads the peer's stale slot.
7. `RunAll`: `comm->CheckLockstep({kRecoveryTag, new_base, calls_ch0, calls_ch1})` -- both ranks
   must report the same counters -- then `comm->SelfTest()` (6.3.9).
8. `state_ = kReady`.

### 2.6 Shutdown (`TpModel::~TpModel()`)

1. If `state_ != kReady`, call `comm->Abort(kAbortShutdown)` on every endpoint (host-side store only).
2. For each rank, wait (bounded **30 s**, polling `Idle()`) until the worker has finished whatever it
   is running; `cmd_` is never written while a rank may still be executing it. Then post the
   teardown closure and wait for it (30 s): it destroys `model` then `comm` (frees its VRAM on the
   right device) and calls `hipDeviceSynchronize()`. Then `~RankWorker` sets `exit_ = true` under
   `mu_`, notifies `cv_`, and joins (30 s). If any wait expires: log `[r4dx-tp] rank <r> did not
   finish in 30 s at shutdown` and `std::quick_exit(3)` rather than free memory a kernel may still
   touch.
3. After both joins, free the pinned mailbox (`hipHostFree`) -- only now, never while a rank may
   still have a kernel in flight (same rule as `tp_bench`'s `g_stuck` leak).

### 2.7 Thread-safety fixes in shared code (needed by two rank threads in one process)

| Site | Today | Change |
|---|---|---|
| `src/model/linear.cpp:83` `PickTuning`'s `static std::unordered_map` | unsynchronized | `static thread_local std::unordered_map` (each rank thread builds its own identical cache; TP=1 unchanged) |
| TP=2 tuning rows | -- | new `src/model/gemm_tuning_table_tp2.inc`, included in `linear.cpp` as `namespace tp2 { #include "gemm_tuning_table_tp2.inc" }` (same `kGemmTuningTable` array name, so `tune_gemm.py --out` writes it unchanged). New `void SetTp2TuningForThisThread(bool)` in `linear.h` sets a `thread_local bool`; `Model::Load` calls it with `true` when `opts.tp.world > 1` (that is the rank thread in `TpModel`, the main thread in `tool_tp_step_bench`). `ResolveTuning` on such a thread scans `tp2::kGemmTuningTable` first (same M-band / group rules), then the main table, then `FallbackTuning`. The cache key gains bit 52 = the flag. At TP=1 the flag is never set, so the main table alone is consulted, byte for byte as today. Reason: the TP=2 key `(w4a16, N=17408, K=5120)` is also the DFlash drafter's `gate_proj`/`up_proj` (`dflash_draft.cpp:475-476`), which has no row today (`FallbackTuning`); adding it to the shared table would change TP=1 DFlash bytes and timing |
| `src/kernels/src/r4dx_kernels.hip:27` `int64_t g_launch_count` | plain `++` from every launcher | `std::atomic<int64_t>` with `fetch_add(1, relaxed)`; `_get`/`_reset` use `load`/`store` |
| `r4dx_topk_lse_f32` module-scope `__device__` scratch (`r4dx_kernels.hip:942-945`) | "one call in flight per process" | add `r4dx_topk_lse_f32_ws(..., int64_t workspace)` + `r4dx_topk_lse_workspace_bytes()`; the kernels take the partial arrays as pointers; the old entry point passes the module globals' addresses (TP=1 bytes unchanged). `Model` uses `_ws` with its own workspace whenever `tp_world > 1` (required for emulation, where both ranks share one device) |
| `GdnControlCache` lazy `hipMalloc` + blocking `hipMemcpy` (`gdn_state.h:146-156`) | first use of each `(T, slot)` key mid-inference | add `Prewarm(int64_t max_T, int32_t slot, int64_t window)` (CuPair 1..max_T, CacheIdx(slot), SidxBase(slot, window), HasInitTrue) and `Freeze()` (a miss after freeze throws `std::logic_error`). `Model::TpWarmup()` calls `Prewarm` before the warm-up forward and `Freeze` after it (2.9 step 9), only when `tp_world > 1` |
| `EpilogueForLayout`'s `static const bool kDisabled` | magic static | already thread-safe; no change |
| Device allocation inside a collective command (6.3.7) | not tracked | new `src/core/include/r4dx/core/tp_alloc_guard.hpp`: `thread_local int g_tp_collective_depth` + `std::atomic<uint64_t> g_tp_collective_allocs`; `TpModel::RunCollective` wraps the closure in a `TpCollectiveScope` (depth++ / depth--); `DeviceBuffer`'s allocating constructor, `Resize` (`device_buffer.hpp:66-73`) and `Free` (`114-118`) do `if (g_tp_collective_depth > 0) { ++g_tp_collective_allocs; log once per call site }` -- never throw (the free path is a destructor). `test_tp_emulation` and `tool_tp_soak` require the counter to be 0 at the end. At TP=1 the depth is always 0: one thread-local load per allocation, no behaviour change |

### 2.8 `TextModel`: the interface `src/cli` and `src/server` use

`src/cli` and `src/server` hold a `std::unique_ptr<r4dx::model::TextModel>` instead of a `Model`.
`TextModel` is the exact surface they call today, with two deliberate substitutions (a device pointer
and a `Container&` cannot mean anything across two devices): `GetContainer()` is replaced by three
accessors, and `EncodeImages` writes an `ImageRows` instead of a `core::DeviceBuffer`.

New file `src/model/model_types.h` (includes only `r4dx/core/device_buffer.hpp`,
`vision/grid_thw` types and `<vector>`; **no** `model.h`, so `model.h` and `text_model.h` can both
include it without a cycle). It holds `ImageSpan`, `ImageRows` and `StepProfile`, all moved out of
`Model`; `model.h` keeps `using ImageSpan = r4dx::model::ImageSpan; using StepProfile =
r4dx::model::StepProfile;` inside `class Model`, so every existing `Model::ImageSpan` /
`Model::StepProfile` spelling still compiles. New file `src/model/text_model.h` includes
`model_types.h` and `model_config.h`, never `model.h`.

```cpp
// ---- src/model/model_types.h ----
namespace r4dx::model {

struct ImageSpan {
  int64_t offset = 0;
  int64_t tokens = 0;
  vision::GridThw grid;
  const uint16_t* embeds = nullptr;   // [tokens, hidden] bf16
  bool embeds_on_host = false;        // NEW: true => host memory, spliced with an H2D copy
};

// Owner of one EncodeImages result. TP=1: device rows (today's DeviceBuffer, same bytes).
// TP: pageable host rows (std::vector), copied to each rank's device at splice time.
class ImageRows {
 public:
  const uint16_t* data() const;       // device or host pointer, see on_host()
  bool on_host() const;
  int64_t rows() const;
  core::DeviceBuffer<uint16_t> dev;   // filled by LocalTextModel
  std::vector<uint16_t> host;         // filled by TpModel
};

struct StepProfile { /* moved verbatim from Model::StepProfile */ };
}  // namespace r4dx::model

// ---- src/model/text_model.h ----
namespace r4dx::model {
struct TpOptions {                    // from --tp* flags (section 9)
  int world = 1;                      // 1 or 2
  enum class Mode { kReal, kEmulate, kNoop } mode = Mode::kReal;
  std::vector<int> devices;           // empty = auto (section 9.2)
  int noop_rank = 0;
  int ar_timeout_ms = 500;            // [10, 1500]
  int ar_nb_small = 4;                // channel 0 blocks
  int ar_nb_large = 4;                // channel 1 blocks
  // Test-only fault injection. Set by tests directly, or by TpModel::Load from the environment
  // variable R4DX_TP_FAULT="<rank>:<n>:<kind>" (so smoke.ps1 can drive the production binaries).
  // Fires ONCE per process, at the n-th AllReduceSumBf16 of `rank` counted from the END of warm-up.
  // kind 0: the endpoint throws std::runtime_error("tp fault injection"); kind 1: it sleeps 700 ms
  // before enqueuing (the peer's kernel times out). Logged loudly at load when armed.
  int fault_rank = -1; int64_t fault_at_allreduce = -1; int fault_kind = 0;
};

struct VramReport { int rank; int device; double used_gib, free_gib, total_gib; };

class TextModel {
 public:
  virtual ~TextModel() = default;
  // ---- identity / capabilities (cached at load; HasVision is safe from any thread) -----------
  virtual const ModelConfig& Config() const = 0;          // GLOBAL config
  virtual const std::string& ModelId() const = 0;
  virtual int64_t ImageTokenId() const = 0;
  virtual int64_t VisionMergeSize() const = 0;            // spatial_merge_size, 2 if no vision
  virtual bool HasVision() const = 0;
  virtual bool MtpEnabled() const = 0;
  virtual bool MtpUsingReducedVocabDraft() const = 0;
  virtual bool DflashEnabled() const = 0;
  virtual int64_t SampledFallbackRows() const = 0;
  virtual int64_t PositionCount() const = 0;
  virtual int64_t NumLoadedLayers() const = 0;            // tool_teacher_forced_logprobs.cpp:141
  virtual int TpWorld() const = 0;
  virtual std::vector<VramReport> Vram() const = 0;
  // ---- sequence state ---------------------------------------------------------------------------
  virtual void Reset() = 0;
  virtual void SetDflashInjectionEnabled(bool enabled) = 0;
  // ---- vision -------------------------------------------------------------------------------
  virtual void EncodeImages(const float* pixel_values, int64_t total_patches,
                            const std::vector<vision::GridThw>& grids, ImageRows* out,
                            vision::VisionEncodeStats* stats = nullptr) = 0;
  // ---- forward (identical contracts to Model's methods of the same name) ----------------------
  virtual std::vector<float> Prefill(const std::vector<int32_t>& token_ids) = 0;
  virtual std::vector<float> PrefillMultimodal(const std::vector<int32_t>& token_ids,
                                               const std::vector<ImageSpan>& images) = 0;
  virtual std::vector<float> DecodeStep(int32_t token_id) = 0;
  virtual int32_t DecodeStepGreedy(int32_t token_id) = 0;
  virtual int32_t DecodeStepSampled(int32_t token_id, const kernels::SampleParams& params,
                                    std::mt19937_64& rng) = 0;
  virtual std::vector<int32_t> DecodeStepMtpGreedy(int32_t token_id, int64_t k) = 0;
  virtual std::vector<int32_t> DecodeStepMtpSampled(int32_t token_id, int64_t k,
                                                    const kernels::SampleParams& params,
                                                    std::mt19937_64& rng) = 0;
  virtual std::vector<int32_t> DecodeStepDflashGreedy(int32_t token_id, int64_t k, float p_min,
                                                      int64_t n_min, int64_t* walk_len_out = nullptr) = 0;
  virtual std::vector<int32_t> DecodeStepDflashSampled(int32_t token_id, int64_t k, float p_min,
                                                       int64_t n_min, const kernels::SampleParams& params,
                                                       std::mt19937_64& rng,
                                                       int64_t* walk_len_out = nullptr) = 0;
  // ---- diagnostics (TP: throws TpUnsupportedError) --------------------------------------------
  virtual StepProfile DecodeStepProfiled(int32_t token_id) = 0;
  virtual StepProfile PrefillProfiled(const std::vector<int32_t>& token_ids) = 0;
};

// world==1 => LocalTextModel(Model::Load(opts)) -- exactly today's Model.
// world==2 => TpModel::Load(opts, tp).
std::unique_ptr<TextModel> LoadTextModel(const ModelOptions& opts, const TpOptions& tp);
}
```

`src/model/local_text_model.{h,cpp}`: `class LocalTextModel final : public TextModel { Model m_; }`,
every method a one-line forward (`ModelId()` -> `m_.GetContainer().ModelId()`, `EncodeImages` ->
`m_.EncodeImages(..., &out->dev, stats)` with `on_host=false`, `Vram()` -> one `hipMemGetInfo` on the
calling thread). No behaviour change.

**Every `Model` call `src/cli` / `src/server` make today, and its TP implementation:**

| Call site today | `TextModel` method | `TpModel` implementation |
|---|---|---|
| `cli/main.cpp:446`, `engine.cpp:106` `Model::Load(opts)` | `LoadTextModel(opts, tp)` | section 2.9 |
| `cli/main.cpp:577` reload on prefix mismatch (`model = Model::Load(opts)`) | TP=1 path unchanged; `tp>1`: `model->Reset()` | `Reset()` (a second full load would need 2x VRAM) |
| `engine.cpp:388` `Reset()` | `Reset()` | recovery if needed (2.5), then `RunAll(m.Reset(); m.SetDflashInjectionEnabled(dflash_injection_))` |
| `engine.cpp:107` `GetContainer().ModelId()` | `ModelId()` | cached from rank 0 at load |
| `cli/main.cpp:489`, `engine.cpp:303` `GetContainer().ImageTokenId()` | `ImageTokenId()` | cached |
| `cli/main.cpp:490-491`, `engine.cpp:305` `GetContainer().Vision().config.spatial_merge_size` | `VisionMergeSize()` | cached |
| `cli/main.cpp:460,490,499`, `engine.cpp:114,192`, `engine.h:115` `HasVision()` | `HasVision()` | cached `rank0.model->HasVision()` (const bool, any thread) |
| `cli/main.cpp:552` `Config().hidden_size` | `Config()` | cached GLOBAL config |
| `engine.cpp:602` `MtpEnabled()` | `MtpEnabled()` | cached |
| `cli/main.cpp:695` `MtpUsingReducedVocabDraft()` | same | cached |
| `cli/main.cpp:608,610` `SampledFallbackRows()` | same | cached from rank 0 after every successful collective command (both ranks count identically; 2.4 host-only table) |
| `engine.cpp:368` `SetDflashInjectionEnabled(b)` | same | host-only: `dflash_injection_ = b` on the facade; applied by `RunCollective` and `Reset()` (2.4). Legal in `kNeedsRecovery` |
| `cli/main.cpp:514`, `engine.cpp:410` `EncodeImages(px, n, grids, &DeviceBuffer, &stats)` | `EncodeImages(..., ImageRows*, stats)` | `RunOne(0)`: `m.EncodeImages(..., &tmp_dev, stats)`, sync, D2H into `out->host` (pageable), `on_host=true` |
| `cli/main.cpp:550-553` builds `vision::ImagePlaceholderSpan{grid, embeds = batch.embeds.data() + row*hidden}`; `cli/main.cpp:595-600` and `engine.cpp:414-419` copy spans into `Model::ImageSpan` field by field | same, via `ImageRows` | `vision::ImagePlaceholderSpan` (`src/vision/image_prompt.h:26`) gains `bool embeds_on_host = false`; the CLI sets it from `batch.embeds.on_host()` and copies it at 595-600; the server sets `ms.embeds_on_host = rows.on_host()` at 414-419. At TP=1 it stays `false` (device rows, today's D2D splice) |
| `tests/model/tool_teacher_forced_logprobs.cpp:141` `GetContainer().NumLoadedLayers()`, `:70,135-137` `hipMemGetInfo` | `NumLoadedLayers()`, `Vram()` | cached; `Vram()` summed over ranks for the tool's `[vram]` line |
| `cli/main.cpp:155`, `engine.cpp:435` `Prefill(tokens)` | same | `RunAll(m.Prefill(tokens))`; each rank returns the full gathered `[248320]` row; compare; return rank 0's |
| `cli/main.cpp:155`, `engine.cpp:436` `PrefillMultimodal(tokens, spans)` | same | `RunAll(m.PrefillMultimodal(tokens, spans))` (spans carry host rows) |
| `engine.cpp:753`, `cli/main.cpp:363` `DecodeStep(tok)` | same | `RunAll(m.DecodeStep(tok))` -> gathered full row |
| `cli/main.cpp:174,331` `DecodeStepGreedy(tok)` | same | `RunAll(m.DecodeStepGreedy(tok))` -> merged token |
| `engine.cpp:783`, `cli/main.cpp:383` `DecodeStepSampled(tok, sp, rng)` | same | rng copies (2.3); `RunAll(m.DecodeStepSampled(tok, sp, rng_r))` |
| `cli/main.cpp:293-294`, `engine.cpp:696-697` `DecodeStepMtp{Greedy,Sampled}` | same | `RunAll` (+ rng copies for sampled) |
| `cli/main.cpp:248-250`, `engine.cpp:637-639` `DecodeStepDflash{Greedy,Sampled}(…, &walk_len)` | same | `RunAll`; compare rounds and `walk_len`; rank 0's `walk_len` |
| `cli/main.cpp:133,176` `PrefillProfiled` / `DecodeStepProfiled` | same | throw `TpUnsupportedError` (the CLI rejects `--profile*` with `--tp 2` first) |
| `cli/main.cpp:46-50` `VramUsedGiB()` (hipMemGetInfo on the main thread), called at `:444` **before** load and `:448` after | `Vram()` | `RunAll(hipMemGetInfo)`; CLI prints one line per rank when `TpWorld() > 1`. The TP=1 CLI keeps calling `VramUsedGiB()` at 444/448 exactly as today; with `--tp 2` the pre-load call at 444 is **skipped** (`if (args.tp == 1)`) because the facade thread must make no HIP call, and the TP load log already prints each rank's free/total before load (2.9 step 3) |

Rejected alternative: templating `RunTurn`/`Engine` on the model type. It duplicates every call site
in two instantiations and still cannot hide `DeviceBuffer`/`Container&` in the signatures.

### 2.9 `TpModel::Load(const ModelOptions& opts, const TpOptions& tp)` sequence

1. **Validate** `tp.world == 2`; resolve `devices` (9.2). `hipGetDeviceCount` on a helper thread.
   **Staged rejections** (9.1) are enforced here too, not only in the arg parsers, so a test or tool
   that builds a `TpModel` directly cannot run a path whose TP hooks are not in yet: until P5,
   `opts.mtp_draft_k > 0`, a non-empty `opts.dflash_container` and `opts.vision == kOn` throw
   `TpUnsupportedError` (`kAuto` loads text-only under TP until P5, with a log line); until P4,
   `tp.mode == kReal` throws. Each phase deletes exactly the rejection it implements.
2. **Spawn** `RankWorker(r, devices[r])` for each rank (noop: one worker, rank `tp.noop_rank`;
   emulate: two workers on the same device).
3. **Probe** (`RunAll`): each rank returns `hipDeviceProp_t` fields (`name`, `gcnArchName`,
   `pciBusID`, `pciDeviceID`, `canMapHostMemory`, `totalGlobalMem`),
   `hipDeviceAttributeWallClockRate`, and `hipMemGetInfo` free bytes. Real mode requires: distinct
   `pciBusID`; identical `gcnArchName` (`gfx1201`); `canMapHostMemory == 1`. Log one line per rank:
   `[r4dx-tp] rank r -> HIP device d (<name>, pci <bus>:<dev>, <free>/<total> GiB)`.
4. **Wall-clock check** (`RunAll`): `r4dx_tp_clock_probe` (port of `clock_probe_kernel`,
   `ar_bench.hip:685-692`) against the host clock; > 5% off -> throw (every timeout depends on it).
5. **Decide embed residency once**: `resident = opts.embed_device_resident` and, for every
   **physical** device (grouped by `pciBusID`), `free_before_load >= 2 * embed_bytes * (ranks on
   that device)` -- the 2x heuristic of `container.cpp:280-295`, applied jointly. In real mode that is
   one rank per device; in emulate mode both ranks sit on device 1, so both mirrors are charged
   against its free memory. Passed to every rank as `tp.embed_device_resident_decided`, so both ranks
   take the same gather path.
6. **Shared host tables**: `RunOne(0)`: `Container::LoadEmbedTokensHost(path)` ->
   `std::shared_ptr<const core::PinnedBuffer<uint16_t>>` (2.37 GiB, pinned ONCE for the process with
   `hipHostMallocPortable`, so rank 1's device can DMA from it too); facade thread, CPU only: if
   `opts.dflash_container` is set, `DflashDraft::LoadHostCodebooks(path)` ->
   `std::shared_ptr<const DflashHostCodebooks>` (2 x 127 MB of plain `std::vector`, once). The facade
   thread itself still makes no HIP call.
7. **Comm group**: `TpGroup::Create(mode, world, geometry, timeout)`; for `kReal` the pinned mailbox
   is allocated by `RunOne(0)` (after that thread's `hipSetDevice`) and **immediately zeroed** in
   the same command (host `memset` of the whole region + `std::atomic_thread_fence(seq_cst)`;
   `hipHostMalloc` does not promise zeroed memory, and a garbage flag "ahead" of the base under the
   wrap-safe compare would be a `PROTOCOL_VIOLATION` on the first all-reduce). Then `RunAll`:
   `rank.comm = group->CreateEndpoint(rank, &worker.Heartbeat())` (per-rank
   `hipHostGetDevicePointer`, VRAM seq counters and `Status`, seq base; the endpoint bumps the
   heartbeat on every `AllReduceSumBf16` and registers it with `HostExchange::SetHeartbeat`).
8. **Load models** (`RunAll`, both in parallel): `Model::Load(rank_opts)` where `rank_opts = opts`
   plus `rank_opts.tp = TpRankOptions{world, rank, comm, shared_embed, resident, rank == 0 /*vision
   weights*/, shared_codebooks}`. An exception on one rank does not affect the other (no collectives
   yet); the facade destroys both and rethrows.
9. **Warm-up** (`RunAll`, one collective command), in this order:
   1. `comm->SelfTest()` (6.3.9), then a latency sample: rank 0 times 200 back-to-back 10 KiB ARs
      with hipEvents; log `[r4dx-tp] all-reduce self-test OK, isolated L(10 KiB) = x.xx us`.
   2. `comm->SetAllReduceTimeoutMs(1500)` (a kernel argument, 6.3.4), then `m.TpWarmup()`: allocate
      the `topk_lse` workspace; `GdnControlCache::Prewarm(max_chunk_, slot, window)`; then run the
      real paths once through the public methods with fixed token ids `0..63` -- `Prefill` of one
      64-row chunk (channel 1), `DecodeStepGreedy` (channel 0), and when enabled one
      `DecodeStepMtpGreedy(tok, mtp_draft_k)` or `DecodeStepDflashGreedy(tok, dflash_draft_k, ...)`
      (verify window, H6/H7). This is the first touch of every libr4d / r4dx code object on each
      rank (lazy module loads), fills each rank's `PickTuning` cache and faults in every weight
      page, all under the relaxed timeout instead of inside the first user request. Then
      `m.Reset()`, `GdnControlCache::Freeze()`, `comm->SetAllReduceTimeoutMs(tp.ar_timeout_ms)`.
      `Prewarm` stays as an assertion that the warm-up covered every key; `Freeze` turns any later
      miss into a `std::logic_error`.
   3. Warm-up uses only greedy methods, so `SampledFallbackRows()` is still 0 afterwards; the
      fault-injection counter starts after this step (see `TpOptions`).
10. **Cache** the global config, model id, image token id, vision merge size, `HasVision`,
    `MtpEnabled`, `MtpUsingReducedVocabDraft`, `DflashEnabled`, `NumLoadedLayers` from rank 0 (and
    assert rank 1 agrees on everything except `HasVision`); `dflash_injection_ = true` (the `Model`
    default, `model.h:872`).

---

## 3. Config split

### 3.1 `ModelConfig` (`src/model/model_config.h`): what stays global, what is per rank

| Field | Global (v6) | Rank r at TP=2 | Rule |
|---|---|---|---|
| `hidden_size` | 5120 | 5120 | global |
| `num_hidden_layers`, `layer_types` | 64 | 64 | global |
| `num_attention_heads` | 24 | **12** | / world |
| `num_key_value_heads` | 4 | **2** | / world |
| `head_dim` | 256 | 256 | global |
| `attn_output_gate`, `partial_rotary_factor`, `rope_theta`, `mrope_interleaved`, `mrope_section` | | | global |
| `intermediate_size` | 17408 | **8704** | / world |
| `linear_num_key_heads` | 16 | **8** | / world |
| `linear_num_value_heads` | 48 | **24** | / world |
| `linear_key_head_dim`, `linear_value_head_dim`, `linear_conv_kernel_dim` | 128, 128, 4 | same | global |
| `rms_norm_eps`, `tie_word_embeddings`, `mtp_num_hidden_layers` | | | global |
| `vocab_size` | 248320 | **248320** | **stays global** (embedding table, sampler, id range checks); the per-rank lm_head width is `VocabShardSize()` / `LmHead().N` |
| NEW `tp_world` | 1 | 2 | |
| NEW `tp_rank` | 0 | r | |
| derived `KeyDim()` / `ValueDim()` / `ConvDim()` | 2048 / 6144 / 10240 | 1024 / 3072 / 5120 | automatic |
| derived `AttnGqa()` / `GqaRepeats()` / `RotaryDim()` | 6 / 3 / 64 | 6 / 3 / 64 | invariant (asserted) |

New members:

```cpp
int tp_world = 1;
int tp_rank = 0;
int64_t VocabShardSize() const { return vocab_size / tp_world; }        // 124160
int64_t VocabShardBegin() const { return tp_rank * VocabShardSize(); }  // 0 or 124160
bool IsShard() const { return tp_world > 1; }
// Returns the rank-local config. Throws std::invalid_argument naming the failing field unless ALL of:
//   world in {1,2}; 0 <= rank < world; tp_world == 1 on `global` (no double sharding);
//   num_attention_heads % world == 0, num_key_value_heads % world == 0, AttnGqa() unchanged;
//   linear_num_key_heads % world == 0, linear_num_value_heads % world == 0, GqaRepeats() unchanged;
//   (num_attention_heads/world*head_dim) % 512 == 0      (row-parallel K of attn.o: 3072)
//   (ValueDim()/world) % 512 == 0                        (row-parallel K of gdn.out_proj: 3072)
//   (intermediate_size/world) % 512 == 0                 (row-parallel K of mlp.down: 8704 = 17*512)
//   every column-parallel rank segment % 16 == 0         (qkv 1024/1024/3072, z 3072, qg 6144, kv 512,
//                                                         gate/up 8704)
//   vocab_size % (16 * world) == 0                       (124160 = 7760 tiles)
//   !tie_word_embeddings
static ModelConfig Shard(const ModelConfig& global, int world, int rank);
```

`%512` is `FallbackTuning`'s requirement (`linear.cpp:33-44`); it also implies every group (64 / 128
/ 32) and every packed block (64 / 16) divides the per-rank K.

### 3.2 Who uses which config

| Consumer | Config | Notes |
|---|---|---|
| `Container` | holds both: `global_config_` (parsed from the file, used for byte strides and slicing) and `config_` = `ModelConfig::Shard(global, world, rank)` (used for `QuantLinear::N/K` and returned by `Config()`) | `Container::GlobalConfig()` is new; at world 1 the two are equal because the TP=1 path sets `config_ = global_config_` and never calls `Shard` (5.1 step 7; `Shard`'s TP-only checks would refuse e.g. a tied-embedding container TP=1 loads today) |
| `GdnLayer`, `Mlp`, `AttentionLayer` (via `MakeAttnConfig`), `GdnStateManager`, `PagedKvCache`, `MtpHead` | rank | run unmodified on the rank config -- only the comm hook is new |
| `Model` | rank config via `container_.Config()`; `Model::GlobalConfig()` new; `vocab_local_ = container_.LmHead().N`, `vocab_offset_ = cfg.VocabShardBegin()` | `Model::Config()` keeps returning `container_.Config()` (rank); `TextModel::Config()` returns global |
| embedding gather (`EmbedTokensDeviceGather` clamp, host `EmbedTokens`) | `vocab_size` (global) | the table is replicated |
| `SampleCanonical`, `Argmax` on gathered rows | global | |
| `DflashDraft` | its own `Dflash2Config` + `lm_head_vocab` = rank shard, `vocab_offset`, `global_vocab` (8.2) | |
| vision | its own `VisionConfig` | rank 0 uploads weights; every rank parses the config |

### 3.3 Options plumbing

`src/model/model.h`, new struct in `ModelOptions`:

```cpp
struct TpRankOptions {
  int world = 1;
  int rank = 0;
  core::TpComm* comm = nullptr;                            // non-owning; nullptr iff world == 1
  std::shared_ptr<const core::PinnedBuffer<uint16_t>> shared_embed_host;  // TP: one host copy
  int embed_device_resident_decided = -1;                  // -1: Container's own heuristic (TP=1)
  bool vision_weights_on_this_rank = true;                 // TP: rank 0 only
  std::shared_ptr<const DflashHostCodebooks> dflash_codebooks;  // TP: one host copy
};
TpRankOptions tp;   // in ModelOptions; default-constructed == today
```

`src/model/container.h`, new load overload (the existing positional `Load` forwards to it with
`tp_world = 1`, so every existing test/tool is unchanged):

```cpp
struct ContainerLoadOptions {
  Layout layout = Layout::kBf16, lm_head_layout = Layout::kBf16, mtp_head_layout = Layout::kBf16;
  int64_t layer_limit = -1;
  bool embed_device_resident = true;
  int embed_device_resident_decided = -1;
  bool load_vision = false;          // upload vision.* weights on THIS rank
  bool parse_vision_config = false;  // parse vision_config even when not uploading (TP rank > 0)
  int tp_world = 1, tp_rank = 0;
  std::shared_ptr<const core::PinnedBuffer<uint16_t>> shared_embed_host;
};
static Container Load(const std::string& path, const ContainerLoadOptions& o);
static std::shared_ptr<const core::PinnedBuffer<uint16_t>> LoadEmbedTokensHost(const std::string& path);
const ModelConfig& GlobalConfig() const;
bool HasVisionConfig() const;                        // weights uploaded OR config parsed
const vision::VisionConfig& VisionCfg() const;       // valid iff HasVisionConfig()
```

`EmbedTokensHost()` returns `shared_embed_host_->data()` when set, else `embed_tokens_.data()`.

---

## 4. Sharding

### 4.1 Rule kinds

- **replicate** -- every rank loads the full tensor.
- **rows(segments)** -- column-parallel (output dim N split). The tensor's N rows are the
  concatenation of fused *segments*; each segment is split into `world` equal contiguous parts; rank
  r takes part r of every segment, concatenated in segment order.
- **cols** -- row-parallel (input dim K split): rank r takes K range `[r*K/world, (r+1)*K/world)`.
- **rank0-only** -- vision tower weights.

### 4.2 Tensor table (text model, MTP head, vision, DFlash2)

`{L}` = one of `bf16.w` (plus the bare pre-R1 form and `--keep-bf16` fallbacks, all bf16),
`w4a16.wq/wsz`, `w4a8.wq/ws`, `mxfp4.wq/ws/wref`. Ranges are global row/col indices; r in {0,1}.

**Text layers** (`text.layers.{i}.`), GDN layers (48):

| Tensor | Forms | Logical shape | TP=2 | Rank r rows / cols | Rank shape |
|---|---|---|---|---|---|
| `input_layernorm`, `post_attention_layernorm` | bf16 | [5120] | replicate | -- | [5120] |
| `gdn.in_proj_qkv.{L}` | all | [10240, 5120] | rows, 3 segments q[0,2048) k[2048,4096) v[4096,10240) | [1024r, +1024) ∪ [2048+1024r, +1024) ∪ [4096+3072r, +3072) | [5120, 5120] |
| `gdn.in_proj_z.{L}` | all | [6144, 5120] | rows | [3072r, +3072) | [3072, 5120] |
| `gdn.in_proj_a`, `gdn.in_proj_b` | bf16 | [48, 5120] | rows | [24r, +24) | [24, 5120] |
| `gdn.conv1d_weight` | bf16 | [10240, 1, 4] | rows, same 3 segments as qkv | [1024r,+1024) ∪ [2048+1024r,+1024) ∪ [4096+3072r,+3072) | [5120, 4] |
| `gdn.A_log`, `gdn.dt_bias` | fp32 | [48] | rows | [24r, +24) | [24] |
| `gdn.norm_weight` | bf16 (widened to fp32 on load) | [128] | replicate | -- | [128] |
| `gdn.out_proj.{L}` | all | [5120, 6144] | cols | K [3072r, +3072) | [5120, 3072] |
| `mlp.gate_up.{L}` | all | [34816, 5120] | rows, 2 segments gate[0,17408) up[17408,34816) | [8704r, +8704) ∪ [17408+8704r, +8704) | [17408, 5120] |
| `mlp.down.{L}` | all | [5120, 17408] | cols | K [8704r, +8704) | [5120, 8704] |

Attention layers (16):

| Tensor | Forms | Logical shape | TP=2 | Rank r | Rank shape |
|---|---|---|---|---|---|
| `attn.qg.{L}` | all | [12288, 5120], per-head interleaved `[q256 | gate256]` per head | rows | [6144r, +6144) = heads 12r..12r+11, each head's 512 rows intact | [6144, 5120] |
| `attn.k.{L}`, `attn.v.{L}` (v6: `.bf16.w` via `--keep-bf16`) | all + bare | [1024, 5120] | rows | [512r, +512) = kv heads 2r, 2r+1 | [512, 5120] |
| `attn.o.{L}` | all | [5120, 6144] | cols | K [3072r, +3072) = heads 12r..12r+11 | [5120, 3072] |
| `attn.q_norm`, `attn.k_norm` | bf16 | [256] | replicate | -- | [256] |
| `attn.k_descale`, `attn.v_descale` | fp32 | [4] | rows | [2r, +2) | [2] |

Head-mapping checks (why contiguous head blocks are correct): attention q head h reads kv head h/6;
rank r's q heads 12r..12r+11 map to kv heads 2r, 2r+1 = its local kv heads, and the local gqa is 6.
GDN v head hv reads k head hv/3; rank r's v heads 24r..24r+23 map to k heads 8r..8r+7 = its local k
heads (`r4d_gdn_recurrent_update...hip:100`, `chunk_scan:261`). The conv kernel maps channel unit u to
q/k/v by `u < Hg`, `u < 2Hg` (`conv:148-149`), which the 3-segment gather above reproduces.

**Global tensors:**

| Tensor | Forms | Logical shape | TP=2 | Rank r | Rank shape |
|---|---|---|---|---|---|
| `text.embed_tokens` | bf16 | [248320, 5120] | replicate (host: ONE pinned copy per process; device mirror per rank) | -- | [248320, 5120] |
| `text.final_norm` | bf16 | [5120] | replicate | -- | [5120] |
| `lm_head.{L}` | all | [248320, 5120] | rows (vocab split) | [124160r, +124160) | [124160, 5120] |

**MTP head** (`mtp.`; `Container::HasMtp()`; loaded whenever the container has `mtp.norm`, as today):

| Tensor | TP=2 |
|---|---|
| `mtp.attn.qg/k/v/o`, `mtp.attn.q_norm/k_norm/k_descale/v_descale`, `mtp.input_layernorm`, `mtp.post_attention_layernorm`, `mtp.mlp.gate_up/down` | exactly the attention-layer / MLP rules above (k/v are bf16 bare or `.bf16.w`) |
| `mtp.fc` | bf16 [5120, 10240], **replicate** (105 MB; K-splitting it would add a third all-reduce per draft step) |
| `mtp.norm`, `mtp.pre_fc_norm_hidden`, `mtp.pre_fc_norm_embedding` | replicate |
| `mtp.draft_head.lm_head.{L}`, `mtp.draft_head.vocab_ids` | **replicate** (small; lets the reduced-vocab draft argmax stay on device with no merge) |

**Vision** (`vision.*`, 333 bf16 tensors, 0.92 GiB): **rank 0 only** (`load_vision = rank == 0`);
every rank parses `model_config.vision_config` (`parse_vision_config = true`) so rank 1 knows
`spatial_merge_size` and that splicing is enabled.

**DFlash2 draft container** (separate file, `dflash.*`): **replicate everything** on every rank --
`dflash.fc.{L}`, `dflash.enc_output_norm`, `dflash.output_norm`, `dflash.selector.hidden.{L}`,
`dflash.layers.{i}.{input_layernorm, post_attention_layernorm, self_attn.{q,k,v,o}_proj.{L},
self_attn.{q,k}_norm, self_attn.conv.{base,proj.{L}}, mlp.{gate,up,down}_proj.{L}, mlp.conv.{base,
proj.{L}}}`. The two host codebooks `dflash.selector.{predecessor,successor}` are read once per process
and shared by both ranks' `DflashDraft`.

### 4.3 Physical byte slicing per layout (the rules `src/model/tp/tp_shard.cpp` implements)

Notation: full logical `W[N, K]`; rank rows = list of ranges `[a, a+n)`; rank cols `[k0, k0+k)`;
`g` = the w4a16 group read from the container's `__metadata__.quant.w4a16.group` (64 for v6; the
loader already checks it against the kernel, `container.cpp:58-68`).

| Part | Physical layout | Row range `[a, a+n)` | Col range `[k0, k0+k)` | Alignment required |
|---|---|---|---|---|
| bf16 `W` (`.bf16.w`, bare, 2-D raw) | row-major `[N, K]`, 2 B | bytes `[a*K*2, (a+n)*K*2)` -- contiguous | N chunks: for each row n, bytes `[(n*K + k0)*2, +2k)` | none |
| `w4a16.wq`, `w4a8.wq` | `(t, kb, lh, r, s)`, 512 B per (16-row tile t, 64-K block kb); `PackW4Nibbles`, `quant_int4.hpp:221` | bytes `[a*K/2, (a+n)*K/2)` -- contiguous (tile-major) | N/16 chunks: for tile t, bytes `[(t*(K/64) + k0/64)*512, +(k/64)*512)` | rows: a, n % 16; cols: k0, k % 64 |
| `w4a16.wsz` | **uint32** per (t, g, r): dword `(t*(K/g) + gi)*16 + r`; `PackW4A16Scales` | dwords `[a*K/g, (a+n)*K/g)` -- contiguous | N/16 chunks: dwords `[(t*(K/g) + k0/g)*16, +(k/g)*16)` | rows % 16; cols % g |
| `w4a8.ws` | **uint32** per (t, g, r) with g = 128 (NOT uint16; `PackW4A8Scales`, `container.cpp:154` `UploadRawU32`) | dwords `[a*K/128, (a+n)*K/128)` | N/16 chunks: dwords `[(t*(K/128) + k0/128)*16, +(k/128)*16)` | rows % 16; cols % 128 |
| `mxfp4.wq` | `(nt, ks, lane)`, 128 B per (16-row tile, 16-K step); `PackMxfp4Wq` | bytes `[a*K/2, (a+n)*K/2)` -- contiguous | N/16 chunks: bytes `[(nt*(K/16) + k0/16)*128, +(k/16)*128)` | rows % 16; cols % 32 |
| `mxfp4.ws` | uint8 `[K/32][N]` (`PackMxfp4Ws`) | K/32 chunks: for each kg, bytes `[kg*N + a, +n)` for every range, concatenated -> `[K/32][n_total]` | bytes `[(k0/32)*N, ((k0+k)/32)*N)` -- contiguous | rows: none; cols % 32 |
| `mxfp4.wref` | int8 `[N]` | bytes `[a, a+n)` | **the FULL `[N]`, unchanged** (see below) | -- |
| 1-D vectors (`A_log`, `dt_bias`, `*_descale` fp32; norms bf16) | `[N]` | elements `[a, a+n)` | -- | -- |
| `conv1d_weight` | bf16 `[conv_dim][4]`, 8 B per row | bytes `[a*8, (a+n)*8)` per range | -- | -- |

Multi-segment row slices are the concatenation of each segment's slice in segment order; for every
tile-major part that is exactly `pack(W[rank rows])` because packing never mixes rows of different
16-row tiles. For `mxfp4.ws` the concatenation happens inside each k-group row.

**The mxfp4 `wref` exception.** `QuantizeMxfp4` stores absolute E8M0 group exponents in `ws` and
`wref = max over the row's groups` (`quant_mxfp4.hpp:79-110`); the kernel dequantizes with `dsh =
clamp(wref - Ws, 0, 15)` and a per-row factor `2^(wref-127)` (`r4d_gemm_mxfp4a8_nt_m64.hip:190-192,
225`). Keeping the **full-row** `wref` for a K-slice reproduces TP=1's dequantized value for every
element bit-for-bit. `pack(slice_K(W)).wref` would be the shard's max instead, which can un-clamp
groups TP=1 clamps. So the loader keeps the full `wref`, and the byte-exact test asserts
`sliced.wref == full.wref` rather than `== pack(slice).wref`.

**Row-parallel activations** (not a slicing rule, a numerics note): for w4a8/mxfp4, `QuantActI8` /
`r4dx_quant_act_fp8e4m3_row` and the fused silu_mul epilogue compute per-row scales over the local K
shard (3072 / 8704). That is deterministic and identical between emulation and real TP, but it moves
w4a8/mxfp4 further from TP=1 than w4a16 moves (w4a16 casts to f16 with no scale).

### 4.4 Per-rank state

| State | TP=1 | Per rank at TP=2 | Notes |
|---|---|---|---|
| Paged fp8 KV cache, per attention layer (16) | `max_blocks x 4 heads x 16 x 512 B` | 2 heads | 262144 ctx: 8.0 GiB -> **4.0 GiB/rank** |
| GDN recurrent state, per GDN layer (48) | `(1 + window) slots x 48 x 128 x 128 x 4 B` | 24 heads | window 8 (DFlash): 1.27 GiB -> **0.63 GiB/rank** |
| GDN conv state | `2 x 10240 x (2 + window) x 2 B` per layer | conv_dim 5120 | small |
| `buf_a_`, `buf_b_`, `buf_normed_*`, `embed_staging_`, `attn_positions_`, arena 96 MiB | full hidden | same (activations are replicated) | |
| `logits_dev_` | `[248320]` fp32 | **`[124160]`** | `vocab_local_` |
| `verify_logits_dev_` | `[W x 248320]` | **`[W x 124160]`** | |
| `argmax_dev_` / `verify_argmax_dev_` | int32 | NEW `argmax_pair_dev_[W x 2]` (int32 idx, float val) under TP | TP=1 keeps the old buffers |
| row-summary scratch | `[W x 64]` ids/vals + lse | same (local ids) + `topk_lse` workspace (`r4dx_topk_lse_workspace_bytes()`) | |
| MTP: `MtpHead` KV cache | 4 heads | 2 heads | sharded head |
| MTP: `MtpHead::logits_dev_` | `[vocab_size]` (`mtp_head.cpp:30`; also receives the reduced draft head's `draft_vocab_size` logits, in bounds only because it is full-vocab sized, `mtp_head.cpp:193-194`) | **`[max(lm_head.N, draft_lm_head.N)]`** | the replicated reduced head is NOT capped at vocab/2 by the container format |
| MTP: `mtp_seed_hidden_`, `mtp_num_accepted_dev_` | replicated values | identical on both ranks | |
| DFlash: whole `DflashDraft` (weights, 5-layer KV ring 2048 slots, scratch) | 2.03 GiB incl. ring | **2.03 GiB on each rank** (replicated) | `logits_dev_ [8 x 124160]` |
| Vision tower + arena | 0.92 GiB + lazy scratch | rank 0 only | |
| TP comm | -- | 2 x `u32[64]` seq + 64 B `Status` VRAM; mailbox is host memory | |

### 4.5 Per-rank VRAM (production, v6 w4a16)

| Component | TP=1 | Rank 0 | Rank 1 |
|---|---|---|---|
| body weights | 13.01 GiB | 6.51 | 6.51 |
| lm_head (4-bit) | 0.67 | 0.33 | 0.33 |
| embedding device mirror | 2.37 | 2.37 | 2.37 |
| MTP head (sharded except fc) | ~0.31 | ~0.20 | ~0.20 |
| vision (`--vision auto`) | 0.92 | 0.92 | 0 |
| DFlash drafter (`--dflash`) | 2.03 | 2.03 | 2.03 |
| KV + GDN state at 262144 ctx, window 8 | ~9.3 | ~4.7 | ~4.7 |
| **total, server config, 262k ctx** | ~28.6 | **~17.1** | **~16.1** |

Device 0 also carries the desktop; ~15 GiB headroom remains on each card.

---

## 5. Loader

### 5.1 `Container::Load(path, ContainerLoadOptions)` with `tp_world > 1`

1. Parse `__metadata__` exactly as today; `global_config_ = ModelConfig::FromJson(text_cfg)`;
   `config_ = ModelConfig::Shard(global_config_, world, rank)`; `CheckQuantGroups` unchanged; read
   `w4a16_group = metadata.quant.w4a16.group` (default 128 when absent -- same rule as
   `CheckQuantGroups`).
2. Open `SafetensorsReader` (mmap) as today. Each rank opens its own mapping; the OS shares pages.
3. `text.embed_tokens`: if `shared_embed_host` is set, **do not** allocate or memcpy a host copy; use
   the shared one. Device mirror: if `embed_device_resident_decided >= 0` use it instead of the
   free-VRAM heuristic; upload from the shared pinned copy.
4. For every tensor, look up `tp::ShardRule` (5.2) from its **base name** (layout suffix stripped),
   then:
   - resolve the on-disk form with the existing `LoadQuantLinearWithFallback` chain (requested ->
     `.bf16.w` -> bare);
   - for each part of that form, compute the rank byte ranges with `tp::PlanRows`/`tp::PlanCols`;
   - if the plan is **one contiguous byte range**, upload straight from the mmap pointer
     (`CopyFromHost(reader.Data(name) + off, count)`) -- no staging copy (qg, z, k/v, lm_head, a/b,
     A_log, dt_bias, descales, mxfp4 wq rows, mxfp4 ws cols);
   - otherwise gather into a per-rank reusable host staging `std::vector<uint8_t>` (grown to the
     largest gathered tensor, ~50 MB) and upload once (qkv, gate_up, conv1d, every K-slice);
   - set `QuantLinear::N/K` to the **rank** shape.
5. MTP: same rules with the `mtp.` prefix; `mtp.fc`, `mtp.*norm*`, `mtp.draft_head.*` replicate.
6. Vision: `load_vision` uploads on rank 0; otherwise, if `parse_vision_config` and the container has
   vision tensors, store `VisionConfig::FromJson(model_config.vision_config)` only.
7. The `tp_world == 1` path is the existing code, untouched (no rule lookup, no staging).
8. Log per rank: `[r4dx::model::Container] rank r/2: <n> tensors sharded, <m> replicated, <b> GiB
   uploaded, <s> GiB gathered through staging`.

Load time: each rank uploads ~half the body plus the replicated tensors, in parallel with the other
rank; expect <= today's 18.6 s.

### 5.2 `src/model/tp/tp_shard.{h,cpp}` (CPU-only; target `r4dx_tp_shard`, no HIP)

```cpp
namespace r4dx::model::tp {
struct Segment { int64_t begin, rows; };             // global rows of one fused segment
struct Range { int64_t begin, count; };
enum class Split { kReplicate, kRows, kCols, kRank0Only };
struct ShardRule { Split split = Split::kReplicate; std::vector<Segment> segments; int64_t k_total = 0; };

// base: container base name without the ".{layout}" suffix (e.g. "text.layers.3.gdn.in_proj_qkv",
// "mtp.attn.o", "lm_head", "text.layers.3.gdn.A_log"). global: the unsharded config.
// Throws std::invalid_argument for a name it does not know (a new tensor must be classified, never
// silently replicated).
ShardRule RuleFor(const std::string& base, const ModelConfig& global);

std::vector<Range> RankRows(const ShardRule& r, int world, int rank);  // concatenation order
Range RankCols(const ShardRule& r, int world, int rank);

enum class Part { kBf16, kW4Wq, kW4a16Wsz, kW4a8Ws, kMxWq, kMxWs, kMxWref, kElem };
struct PartShape {
  Part part; int64_t N = 0, K = 0;
  int group = 0;            // kW4a16Wsz: container w4a16 group; kW4a8Ws: 128; kMxWs: 32
  int64_t row_bytes = 0;    // kElem: bytes per row (e.g. 8 for conv1d, 4 for fp32 vectors)
};
struct ByteRun { size_t src_off, bytes; };           // gather = concat of runs
std::vector<ByteRun> PlanRows(const PartShape&, const std::vector<Range>& rows);  // throws on misalignment
std::vector<ByteRun> PlanCols(const PartShape&, Range cols);                      // kMxWref -> full
std::vector<uint8_t> Gather(const uint8_t* full, size_t full_bytes, const std::vector<ByteRun>&);
}
```

`RuleFor` is a table keyed by the suffix after `text.layers.{i}.` / `mtp.`:
`gdn.in_proj_qkv`, `gdn.conv1d_weight` -> rows `{q KeyDim, k KeyDim, v ValueDim}`;
`gdn.in_proj_z` -> rows `{ValueDim}`; `gdn.in_proj_a|b`, `gdn.A_log|dt_bias` -> rows
`{linear_num_value_heads}`; `gdn.out_proj` -> cols `ValueDim`; `attn.qg` -> rows
`{2*num_attention_heads*head_dim}`; `attn.k|v` -> rows `{num_key_value_heads*head_dim}`;
`attn.k_descale|v_descale` -> rows `{num_key_value_heads}`; `attn.o` -> cols
`num_attention_heads*head_dim`; `mlp.gate_up` -> rows `{intermediate, intermediate}`; `mlp.down` ->
cols `intermediate`; `lm_head` -> rows `{vocab_size}`; `*layernorm`, `*norm*`, `attn.q_norm|k_norm`,
`gdn.norm_weight`, `text.final_norm`, `text.embed_tokens`, `mtp.fc`, `mtp.draft_head.*` -> replicate;
`vision.*` -> rank0-only.

### 5.3 Host memory

| Item | TP=1 | TP=2 process |
|---|---|---|
| `text.embed_tokens` pinned copy | 2.37 GiB | **2.37 GiB, once** (shared_ptr) -- MUST NOT be duplicated |
| DFlash codebooks (host bf16) | 254 MB | **254 MB, once** (shared_ptr) |
| gather staging | -- | ~50 MB per rank during load, freed after |
| mailbox | -- | ~3.3 MB pinned |
| mmapped container | virtual | two mappings, shared page cache |

---

## 6. `TpComm`

### 6.1 Interface -- `src/core/include/r4dx/core/tp_comm.hpp` (header-only, pure virtual)

It lives in `src/core` so `r4dx_model_attention` (header-only `AttentionLayer`, which cannot link
`r4dx_model`) can call it through the interface.

```cpp
namespace r4dx::core {
struct TpError : std::runtime_error { using std::runtime_error::runtime_error; };
struct TpAbortedError : TpError { using TpError::TpError; };     // group poisoned
struct TpTimeoutError : TpError { using TpError::TpError; };     // host-side wait expired
struct TpDivergenceError : TpError { using TpError::TpError; };  // ranks disagreed
struct TpStateError : TpError { using TpError::TpError; };       // call Reset() first / fatal
struct TpUnsupportedError : TpError { using TpError::TpError; };

enum TpAbortCode : uint32_t { kAbortNone = 0, kAbortTimeout = 1, kAbortProtocol = 2,
                              kAbortHost = 3, kAbortShutdown = 5 };

struct TpCommStats {
  uint64_t ar_calls[2] = {0, 0}, ar_bytes[2] = {0, 0};  // per channel
  uint64_t host_exchanges = 0; double host_exchange_wait_us_max = 0;
  uint64_t aborts = 0;
};

class TpComm {
 public:
  virtual ~TpComm() = default;
  virtual int Rank() const = 0;
  virtual int World() const = 0;
  // In place: buf[i] = bf16_rne(f32(buf_rank0[i]) + f32(buf_rank1[i])) on every rank, bit-identical
  // across ranks. Enqueued on `stream`. HostMailbox: never blocks the host. n*2 must be a multiple of
  // 16 and <= MaxAllReduceBytes(). Every rank must make the same sequence of calls with the same n.
  virtual void AllReduceSumBf16(uint16_t* buf, int64_t n, hipStream_t stream) = 0;
  // Blocking host rendezvous: out[r*bytes .. (r+1)*bytes) = rank r's `mine`, for every r, in rank
  // order. Same `bytes` on every rank. Call only with this rank's device work synchronized.
  virtual void HostAllGather(const void* mine, size_t bytes, void* out) = 0;
  // Throws TpDivergenceError if `fingerprint` differs between ranks (one HostAllGather of 32 B).
  virtual void CheckLockstep(const uint64_t fingerprint[4]) = 0;
  // Throws TpAbortedError if any rank aborted (reads host ABORT words / exchange flag; cheap).
  virtual void CheckHealthy() = 0;
  virtual void Abort(uint32_t code, const std::string& why) noexcept = 0;
  virtual bool Aborted() const noexcept = 0;
  virtual size_t MaxAllReduceBytes() const = 0;           // 655360
  virtual void SelfTest() = 0;                            // 6.3.9; throws on failure
  virtual TpCommStats Stats() const = 0;
  // Spin timeout for later AllReduceSumBf16 calls (a kernel argument; no device work). Warm-up
  // raises it to 1500 ms and restores the configured value (2.9 step 9). Clamped to [10, 1500].
  virtual void SetAllReduceTimeoutMs(int ms) = 0;
  // Recovery only (2.5 step 6), with no call of this endpoint in flight: zero every host-side
  // per-endpoint counter (call index, fault counter) and, for EmulatedComm, the xchg buffers.
  virtual void ResetCounters() = 0;
  // Calls per channel since the last ResetCounters(); both ranks must agree (2.5 step 7).
  virtual std::array<uint64_t, 2> CallCounts() const = 0;
};
}
```

Implementations (target `r4dx_tp`, `src/model/tp/`): `NoopComm`, `EmulatedComm`, `HostMailboxComm`,
created by `TpGroup` (`src/model/tp/tp_group.{h,cpp}`), which owns the shared state (mailbox region,
`HostExchange`, seq base, recovery).

### 6.2 Call sites

**Device all-reduce (the only three):**

| # | File / function | Where exactly | Buffer, n |
|---|---|---|---|
| A1 | `src/model/gdn_layer.cpp` `GdnLayer::Forward` | after `ApplyLinear(... w_.out_proj, out_core, gdn_out, T)` (line 237), before the `gdn.residual` block (line 239) | `gdn_out`, `T*hidden` |
| A2 | `src/model/attention/include/r4dx/model/attention/attention_layer.hpp` `AttentionLayer::Forward` | after `ApplyLinear(stream, arena, *w.o, gated, o_out, T)` (line 309), before `attn.residual` (line 313) | `o_out`, `T*hidden` |
| A3 | `src/model/mlp.cpp` `Mlp::Forward` | after the `gemm:mlp.down` `ApplyLinear(... w_.down, h, down_out, T, ...)` block (lines 109-113), before `mlp.residual` (line 115) | `down_out`, `T*hidden` |

Each is `if (comm_ != nullptr) { ProfiledCall(prof, s, "tp.allreduce", [&] { comm_->AllReduceSumBf16(buf, T * hidden, s); }); }`.
The pointer arrives through constructors: `GdnLayer(cfg, input_layernorm, w, core::TpComm* comm =
nullptr)`, `Mlp(cfg, post_attention_layernorm, w, core::TpComm* comm = nullptr)`,
`attention::AttnConfig::comm` (new field, default `nullptr`) set by `MakeAttnConfig(cfg, comm =
nullptr)`. Every construction in `model.cpp` (RunChunk 589/612/634, VerifyWindow 1497/1508/1537, the
two profiled loops) and in `mtp_head.cpp` (ctor `attn_layer_`, Draft's `Mlp`) passes `comm_`. The
R3/P2 fusions are untouched: the fused `x_normed_out` and quant epilogue are computed from the
post-all-reduce, replicated residual.

Per decode token: 48 (A1) + 16 (A2) + 64 (A3) = **128**. MTP draft step: 2 (A2, A3).
`MtpHead::PrimeKv(n)`: **0**. It runs only the attention sublayer and discards its output
(`mtp_head.cpp:337-347`; only the K/V write matters), so an all-reduce there would sum partials
that are thrown away. `MtpHead` therefore owns a second `AttentionLayer prime_attn_layer_` built
with `MakeAttnConfig(cfg, /*comm=*/nullptr)` (an `AttentionLayer` holds only its `AttnConfig`,
`attention_layer.hpp:45,334`), and `PrimeKv` uses it. The K/V projections are column-parallel and
local, so each rank's KV write is complete without communication. Both ranks make the same choice
at construction, so lockstep is unaffected; at TP=1 `comm_` is null anyway, so the two layers are
identical and TP=1 is byte-for-byte unchanged. (RunChunk makes up to two `PrimeKv` calls per chunk,
`model.cpp:692,704`, and `DecodeStepMtp*` one per round, `model.cpp:1729`; with this rule none of
them costs an all-reduce.)

**Host exchange (TP only; every one runs after the rank's own stream synchronize):**

| # | Site | Payload per rank |
|---|---|---|
| H1 | `Model::RunChunk` entry: `CheckLockstep({kind, T, pos_, fnv1a(tokens)})` + `CheckHealthy()` | 32 B |
| H2 | `Model::VerifyWindow` entry: same | 32 B |
| H3 | `RunChunk` end: greedy pair / summary / full-row gather (7.3-7.5) | 8 B / 536 B / 496,640 B |
| H4 | `VerifyWindow` end: T pairs + (sampled) T summaries in ONE gather; when `logits_out != nullptr`, T per-row `GatherVocabRow` calls producing `[T, 248320]` in global id order. **`logits_out` is a production path**: `VerifyAndResolveRound` requests it for every sampled speculative round with `0 < T < kMinSummaryTemperature` (`model.cpp:1621-1632`) and samples from it with the global stride (`model.cpp:1648`) | T x 8 B (+ T x 536 B) (+ T x 496,640 B) |
| H5 | `ReadVerifyLogitsRow`, `DecodeStepSampled` fallback: full-row gather | 496,640 B |
| H6 | `MtpHead::Draft`, full-vocab head only: per draft step pair merge (8.1) | 8 B |
| H7 | `DflashDraft::DraftRound` after its staging D2H: per-row top-16 merge (8.2) | 8 rows x 16 x 8 B = 1 KiB |

### 6.3 `HostMailboxComm` -- the real 2-GPU transport (port of `tools/tp_bench` `flag(d3,a0)`)

#### 6.3.1 Kernel and C ABI

New files `src/kernels/include/r4dx/kernels/tp_kernels.h` and `src/kernels/src/r4dx_tp_kernels.hip`
(second hipcc custom command in `src/kernels/CMakeLists.txt`, linked into `r4dx_kernels`; **same
flags as `r4dx_kernels.hip`: WGP mode, never `-mcumode`** -- the drain-3 argument depends on the WGP
barrier lowering, see `tools/tp_bench/README.md` "drain 3 note").

Port **verbatim** from `tools/tp_bench/ar_bench.hip` (only renaming):

| ar_bench | engine | lines |
|---|---|---|
| `ld_acq_sys`, `ld_rlx_sys`, `st_rel_sys`, `st_rlx_sys`, `now_ticks` | same, `static __device__` | 574-589 |
| `add_bf16x2`, `add_bf16x8`, `f2bf` (RNE) | same; **shared** with `r4dx_tp_add_bf16` (emulation) | 546-551, 599-611 |
| `ld_sticky`, `mark_sticky`, `note_failure`, `note_abort_exit` | same | 614-642 |
| `spin_flag_ge` (s_sleep, wrap-safe `(int)(v - want) >= 0`, abort words every 16 polls, wall_clock64 timeout) | same | 651-676 |
| `ar_entry` (seq bump + sticky check) | same | 791-802 |
| `ar_flag_kernel` | `r4dx_tp_ar_flag_bf16_kernel`, **trace code removed** | 804-893 |
| `Status` (64 B) | `R4dxTpStatus`, identical layout | 203-220 |
| `clock_probe_kernel`, `publish_kernel` | `r4dx_tp_clock_probe`, `r4dx_tp_publish_status` | 685-692, 728-732 |

```c
struct R4dxTpArArgs {                    // passed by value to the kernel
  int64_t buf;                           // uint16_t* bf16, in place (push reads it, reduce rewrites it;
                                         // each 16-B word is read and written by the same thread)
  int64_t peer_mbox, my_mbox;            // this channel's receive mailboxes (device view of host memory)
  int64_t peer_flags, my_flags;          // u32*, one 128-B line per block
  int64_t my_abort, peer_abort;          // u32* in host memory
  int64_t seq;                           // u32[64] VRAM, this channel
  int64_t status;                        // R4dxTpStatus* VRAM
  int32_t n16, nb, w;                    // payload 16-B words, blocks, words per block
  uint32_t blk_stride, slot_stride;      // bytes; blk_stride % 128 == 0
  int32_t drain, acq, spin_acq;          // production: 3, 0, 0
  uint64_t timeout_ticks;                // timeout_ms * WallClockRate(kHz)
};
void r4dx_tp_ar_flag_bf16(const R4dxTpArArgs* a, int nt, int64_t stream);   // grid nb, block nt=256
void r4dx_tp_add_bf16(int64_t inout, int64_t peer, int64_t n, int64_t stream);  // emulation
void r4dx_tp_clock_probe(int64_t out_u64x2, int iters, int64_t stream);
void r4dx_tp_publish_status(int64_t status_dev, int64_t mirror_host, int64_t stream);
```

Algorithm per block b (unchanged from tp_bench): `s = ++seq[b]` (sticky -> skip whole call);
`slot = s & 1`; push this block's words of `buf` into `peer_mbox[slot][b]`; drain (`s_wait_storecnt 0`);
`__syncthreads`; tid0 release-stores `peer_flags[b] = s` (system scope, its `global_wb scope:SYS`
writes back every wave's pushed lines); tid0 spins until `(int)(my_flags[b] - s) >= 0`, checking both
abort words every 16 polls and `wall_clock64` against `timeout_ticks`; `>= s+2` with no abort set is
`PROTOCOL_VIOLATION`; `__syncthreads`; reduce `buf[k] = add_bf16x8(buf[k], my_mbox[slot][b][k])`.
`add` is `f(a)+f(b)` in fp32, one RNE rounding: commutative, so both ranks write identical bits.

#### 6.3.2 Channels and mailbox geometry

Two independent channels, each with its own flags, mailbox slots and VRAM seq counters, so different
message sizes never share a block region (the slot-reuse proof of `ar_bench.hip:86-91` is per block
and needs a fixed region per block):

| Channel | Used when | Max rows | Max bytes | nb (default) | blk_stride | slot_stride | mailbox per rank (2 slots) |
|---|---|---|---|---|---|---|---|
| 0 | `bytes <= 174,080` (decode T=1, DFlash verify T<=8, MTP verify T<=17) | 17 | 174,080 | 4 (`--tp-ar-nb`) | `round_up(ceil(10880/nb)*16, 128)` = 43,520 | 174,080 | 348,160 |
| 1 | `174,080 < bytes <= 655,360` (prefill chunks 18..64 rows, MTP K>16) | 64 | 655,360 | 4 (`--tp-ar-nb-large`, P3 sweeps 4/8/16) | 163,840 at nb=4 | 655,360 | 1,310,720 |

`w = ceil(n16 / nb)` per call (block b owns words `[b*w, min((b+1)*w, n16))`), `n16 = T * 640`.
Channel choice and `nb` depend only on `bytes`, which both ranks compute from the same T.

Shared region (one `hipHostMalloc(Coherent | Mapped | Portable)`, allocated on rank 0's thread; each
rank uses its own `hipHostGetDevicePointer` result -- measured equal to the host pointer, but use the
returned value):

```
0x000000  ABORT[0]           128-B line, written only by rank 0 (device on timeout, host on error)
0x000080  ABORT[1]           written only by rank 1
0x000800  MIRROR[0]          64 B Status mirror of rank 0 (written by rank 0's device)
0x000C00  MIRROR[1]
0x001000  FLAGS[ch0][rank0]  64 lines x 128 B = 8 KiB, written ONLY by rank 1
0x003000  FLAGS[ch0][rank1]  written ONLY by rank 0
0x005000  FLAGS[ch1][rank0]
0x007000  FLAGS[ch1][rank1]
0x010000  MBOX[ch0][rank0]   2 slots, 64 KiB-aligned size (0x60000), written ONLY by rank 1
0x070000  MBOX[ch0][rank1]
0x0D0000  MBOX[ch1][rank0]   (0x140000)
0x210000  MBOX[ch1][rank1]
0x350000  end  (~3.3 MB)
```

No 128-B line is ever written by both devices (`hip_runtime_api.h:4868-4875`). The offsets above are
for the default `nb`; `TpGroup` computes every offset from the geometry (`blk_stride`, `slot_stride`,
each mailbox rounded up to 64 KiB) so a different `--tp-ar-nb*` only moves the boundaries.

#### 6.3.3 Seq counters and session base

Per rank, per channel: `u32 seq[64]` in VRAM, zeroed then `hipMemsetD32Async(seq, base, 64)`. Initial
`base = 0x00001000`; every recovery moves it above every value used (2.5). Stale flags from an older
session compare "behind" under the wrap-safe compare even if a host `memset` was not snooped. 2^32
calls per block at 55 tok/s x 128 ARs is ~7 days; the wrap-safe compare makes the wrap harmless.

#### 6.3.4 `AllReduceSumBf16(buf, n, stream)`

```
bytes = n * 2;  require bytes % 16 == 0 && 0 < bytes <= 655360   (else throw std::invalid_argument)
ch = bytes <= 174080 ? 0 : 1
if (fault injection armed and ++calls == fault_at) { throw / sleep 700 ms }   // tests only
args = channel geometry + {buf, n16 = bytes/16, w = ceil(n16/nb), timeout_ticks, drain 3, acq 0, spin_acq 0}
r4dx_tp_ar_flag_bf16(&args, 256, stream)          // async, no host sync
stats.ar_calls[ch]++, stats.ar_bytes[ch] += bytes
```

#### 6.3.5 Timeout, sticky abort, host detection

- Timeout: `--tp-ar-timeout-ms`, default **500**, clamped to [10, 1500] (TDR is 2 s).
- A block that times out release-stores `ABORT[rank] = kAbortTimeout`, sets VRAM `sticky`, records
  block/seq/phase in `Status`; every later AR of that rank skips entirely (no push), so a stalled
  peer's unread slot is never overwritten.
- `Abort(code, why)` from the host: `HostU32Store(&ABORT[rank], code)` + `HostExchange::Abort(why)`.
  The peer's spinning kernel sees it within 16 polls. The engine builds as **C++17**
  (`CMakeLists.txt:10`, `src/kernels/CMakeLists.txt:25`), while tp_bench used C++20's
  `std::atomic_ref` (`ar_bench.hip:1471-1472`, `build.ps1:26`), so the host side uses two helpers in
  `tp_comm_host_mailbox.cpp` (compiled by clang-cl, which provides the GCC builtins):
  `inline uint32_t HostU32Load(const volatile uint32_t* p) { return __atomic_load_n(p,
  __ATOMIC_ACQUIRE); }` and `inline void HostU32Store(volatile uint32_t* p, uint32_t v) {
  __atomic_store_n(p, v, __ATOMIC_RELEASE); }`. No C++20 construct is ported.
- **Codegen check** (the drain-3 argument depends on it, `tools/tp_bench/README.md` "drain 3 note"):
  `src/kernels/CMakeLists.txt` adds a custom target `r4dx_tp_kernels_isa` that runs hipcc with the
  same flags plus `-S` on `r4dx_tp_kernels.hip`, and a CMake script
  (`src/kernels/check_tp_isa.cmake`) that fails the build unless the `r4dx_tp_ar_flag_bf16_kernel`
  function body contains `global_wb scope:SCOPE_SYS` before the flag's release store and the build
  is not in CU mode (no `-mcumode` in the flags). Built by default; ~40 LOC.
- `CheckHealthy()`: acquire-load `ABORT[0]`, `ABORT[1]`, and the exchange flag. Nonzero -> sync this
  rank's streams (watchdog), D2H the VRAM `Status`, throw
  `TpAbortedError("tp: rank <r> <code> at channel <c> block <b> seq <s> phase <p>; peer abort word
  <x>")`. Called at H1/H2 and after every command (2.2).

#### 6.3.6 Why the sequences stay in lockstep

The kernel pairs calls by per-block sequence number, so both ranks MUST issue the same number of
all-reduces per channel in the same order. That holds because:

- **L1** One facade thread issues every command; both ranks get identical arguments.
- **L2** Every data-dependent branch inside a command reads only replicated or merged values: `pos_`,
  `started_`, `mrope_*`, `draft_window_`, chunk boundaries (`Prefill` splits by `max_chunk_ = 64`),
  verify width `T = candidates.size()`, accepted counts (from merged argmax / merged summaries),
  DFlash drafts (merged top-16 + replicated gate), MTP drafts (merged per step), sampled fallbacks
  (decided from the merged summary). Enforced by review + H1/H2 fingerprints.
- **L3** No rank-specific code inside a collective command. `EncodeImages` is a solo command with no
  collectives; vision weights are the only rank-asymmetric state.
- **L4** No lazy device allocation or free inside a collective command after warm-up
  (`GdnControlCache::Freeze`, the allocation guard of 2.7).
- **L5** Channel and `nb` are functions of the byte count only.
- **L6** Any exception on any rank aborts the group (2.4); `Reset()` rebases the seq counters (2.5),
  so a mid-command failure can never leave the two ranks' counters offset.
- **L7** Results and rng states are compared after every command (2.3).

A lockstep bug therefore shows up as `TpDivergenceError` (H1/H2 or 2.3) or, worst case, as a 500 ms
all-reduce timeout -- never as a hang past 500 ms or silently wrong output.

#### 6.3.7 Implicit-sync hazards (checked, and why they are safe)

- `hipMalloc`/`hipFree`/`hipHostFree` can imply a device sync. **Rule: no device allocation or free
  inside a collective command after warm-up; solo commands may allocate and free.** Collective paths
  use the arena, the frozen control cache and persistent buffers. `EncodeImages` is the one known
  allocating path: it frees a temporary `DeviceBuffer` and lazily grows/reallocates the vision
  arena and output buffer (`src/vision/vision_tower.cpp:109, 364`). That is safe only because it
  is a **solo** command on rank 0 while rank 1 is idle and no all-reduce is in flight; encoding must
  never move into a collective command. The allocation guard (2.7) counts any violation, and the
  TP tests require the count to be 0. `ImageRows` uses pageable host memory, so releasing it on
  the facade thread is not a HIP call.
- Blocking null-stream `hipMemcpy` (RunChunk's `attn_positions_`/`attn_seqused_k_`, `FetchRowSummaries`,
  `CommitVerifiedWindow`) runs only after this rank's `stream_.Synchronize()`, when none of its
  all-reduces is in flight; `stream_` is `hipStreamNonBlocking` so the null stream does not wait on it.

#### 6.3.8 Mailbox sizing for the largest messages

Prefill chunk: 64 rows x 5120 x 2 B = 655,360 B = 640 KiB (channel 1). Verify: DFlash k=7 -> 8 rows =
80 KiB; MTP K=16 -> 17 rows = 170 KiB (channel 0); MTP K up to 63 -> channel 1. Everything fits.

#### 6.3.9 `SelfTest()` (load time and after every recovery)

For each channel and for n in {1, 8, 17 rows} (ch0) / {18, 64 rows} (ch1), 16 iterations k: rank r
fills its buffer with `hipMemsetD16Async(bf16(r == 0 ? k % 64 : 64 + k % 64))`, all-reduces, D2H,
checks every element == `bf16(64 + 2*(k % 64))` (exact; integers < 256), then `CheckHealthy()`.

### 6.4 `NoopComm` (single rank, `--tp-mode noop`)

`AllReduceSumBf16`: no-op (no kernel -- the baseline that `L_vs_no_ar_kernel` is defined against,
1.4). `HostAllGather`: copies `mine` into every slot. `CheckLockstep`/`CheckHealthy`: no-ops.
Loads ONE rank's shard (world 2) on one device, to time the per-rank step. Tokens are meaningless;
only timing is. `NoopComm` is created directly (`tp::MakeNoopComm(world, rank)`,
`src/model/tp/tp_comm_noop.{h,cpp}`) -- it needs no `TpGroup`, so P2a can use it with a bare `Model`
before any threading code exists.

### 6.5 `EmulatedComm` (two ranks, one device, `--tp-mode emulate`)

Both rank threads `hipSetDevice` to the same device; each keeps its own streams. Per call c:

1. `hipMemcpyAsync(xchg[rank][c & 1], buf, bytes, D2D, stream)`; `hipStreamSynchronize(stream)`.
2. `HostExchange::Barrier()` (with abort/timeout).
3. `r4dx_tp_add_bf16(buf, xchg[peer][c & 1], n, stream)` -- the SAME `add_bf16x8` as the mailbox
   kernel, same operand order (mine, peer).

No second barrier: rank r rewrites `xchg[r][p]` only at call c+2, after passing barrier c+1, which
needs the peer to have synchronized its stream at c+1 -- i.e. after the peer's call-c add finished.
`xchg[r][0|1]` are `DeviceBuffer<uint16_t>` of 655,360 B each, allocated at endpoint creation.
The call index `c` is per endpoint and is zeroed by `ResetCounters()` during recovery (2.5 step 6),
together with both `xchg` buffers.
All other behaviour (HostExchange, CheckLockstep, Abort) is shared with `HostMailboxComm`, so an
emulated run executes the same Model code, the same kernels, the same tunings and the same fp32 add
as a real run -- which is why G6 can demand byte identity.

"Sequential" here means: device work is serialized at every all-reduce (both ranks synchronize, then
add), not that rank 0 runs its whole forward before rank 1. The latter is impossible without fibers:
rank 0 cannot finish layer 0 without rank 1's layer-0 partial.

### 6.6 `HostExchange` -- `src/core/include/r4dx/core/tp_host_exchange.hpp` (header-only, CPU)

```cpp
class HostExchange {
 public:
  HostExchange(int world, std::chrono::milliseconds timeout = std::chrono::seconds(30));
  void AllGather(int rank, const void* mine, size_t bytes, void* out);  // rank order
  void Barrier(int rank);
  void Abort(const std::string& why) noexcept;   // wakes every waiter with TpAbortedError
  bool Aborted() const noexcept;
  void Reset();                                  // only with no rank inside
  void SetHeartbeat(int rank, std::atomic<uint64_t>* hb);  // optional; bumped on wait entry/exit
 private:
  // per rank: std::atomic<uint64_t> arrive[r]; uint64_t gen_[r]; std::vector<uint8_t> slot[2][r];
  //           std::atomic<uint64_t>* heartbeat_[r] (nullptr = none)
};
```

Call g of rank r: `g = ++gen_[r]` (the generation counter lives in `HostExchange`, one per rank, so
`Reset()` zeroes it together with `arrive[]`); write `slot[g & 1][r]` (resize if needed),
`arrive[r].store(g, release)`; wait until every `arrive[q] >= g` (spin with `_mm_pause` up to 50 ms,
then `yield`/`Sleep(0)` loop, abort check every iteration, timeout -> `Abort` + `TpTimeoutError`);
copy all slots `[g & 1][*]` to `out`. Slot reuse is safe for the same parity argument as 6.5.
`Reset()` also zero-fills both slot generations and clears the abort flag. Every wait bumps the
calling rank's heartbeat on entry and exit (2.2 step 6).

---

## 7. Output head and sampling

### 7.1 Decision: vocab-split lm_head

| | Replicate (715 MB per rank) | **Vocab-split (357 MB per rank)** |
|---|---|---|
| plain decode lm_head | 1.07 ms/token on each rank | ~0.55 ms (tuned row for N=124160) -> **-0.5 ms/token (+2.8% tok/s)** |
| DFlash verify head (8 rows) | ~1.1 ms | ~0.55 ms |
| DFlash drafter head (8 rows) | ~1.1 ms | ~0.55 ms -> **-1.1 ms/round (+4.5% tok/s)** |
| VRAM per rank | +0.33 GiB | -- |
| merges | none | greedy 8 B/row; sampled 536 B/row; unresolved sampled rows and `Prefill`/`DecodeStep` gather 496,640 B/rank |
| host cost of merges | 0 | one ~2-10 us rendezvous per step (already lockstepped by 128 ARs) |

Split wins on every production path; the merges are exact (7.3-7.4) and CPU-testable.

### 7.2 Model changes

`Model` gains `int64_t vocab_local_` (= `LmHead().N`) and `int64_t vocab_offset_` (=
`cfg.VocabShardBegin()`). At TP=1 `vocab_local_ == vocab_size` and every TP branch is skipped.

**The rule** (the line list below is a checklist, the rule is what must hold): every buffer size,
kernel `vocab` argument, row stride or D2H count of `logits_dev_`, `verify_logits_dev_` or the
row-summary scratch uses **`vocab_local_`**; every host row that `SampleCanonical`, `Argmax` or a
caller sees is a **gathered full row** (`GatherVocabRow`, 7.5) and uses **`cfg.vocab_size`**. No
host code ever reads `vocab_size` floats out of a device logits buffer under TP.

| `model.cpp` line(s) at `aa54c20` | What | Under TP |
|---|---|---|
| 209, 254 | `logits_dev_`, `verify_logits_dev_` allocation | `vocab_local_` |
| 731-733, 1558-1561 | `r4dx_argmax_f32(..., cfg.vocab_size)` | `r4dx_argmax_val_f32(..., vocab_local_)` + pair merge (7.3) |
| 1024, 1056 | `LaunchRowSummaries` / `FetchRowSummaries` `vocab` | `vocab_local_` (+ id offset, merge, 7.4) |
| 766-767 | `RunChunk` full-logits readback (`logits.resize(cfg.vocab_size)` + `logits_dev_.CopyToHost`) | `GatherVocabRow(logits_dev_.data(), logits.data())` into a `cfg.vocab_size` row |
| 1129-1132 | `DecodeStepSampled` fallback: one `vocab` variable used for the D2H **and** `SampleCanonical` | split: D2H + gather through `GatherVocabRow` (local count `vocab_local_`), `SampleCanonical(row, cfg.vocab_size, ...)` |
| 1579-1580 | `VerifyWindow` `logits_out` resize + D2H | T x `GatherVocabRow`, `logits_out` sized `T * cfg.vocab_size` (H4) |
| 1597-1601 | `ReadVerifyLogitsRow` | `GatherVocabRow(verify_logits_dev_.data() + row * vocab_local_, out)` |
| 1095, 1113, 1634, 1648 | `SampleCanonical` over a host row | unchanged: `cfg.vocab_size` over the gathered row |

`src/kernels`: add `r4dx_argmax_val_f32(logits, out_idx, out_val, vocab, stream)` -- the same
`ArgmaxKernel` (`r4dx_kernels.hip:616`) with an extra nullable `float* out_val` written by thread 0;
`r4dx_argmax_f32` passes `nullptr` (TP=1 unchanged).

`src/model/tp/tp_vocab.h` (header-only, HIP-free, CPU-tested):

```cpp
namespace r4dx::model::tp {
struct ArgmaxPair { int32_t idx; float val; };           // idx already GLOBAL
int32_t MergeArgmax(const ArgmaxPair* per_rank, int world);
kernels::RowSummary MergeRowSummaries(const kernels::RowSummary* per_rank, int world,
                                      int64_t global_vocab);   // ids already GLOBAL
void MergeTop16(const int32_t* ids, const float* vals, int world, int rows,   // [world][rows][16]
                int32_t* out_ids, float* out_vals);                          // [rows][16]
double LogAddExp(double a, double b);
}
```

### 7.3 Greedy (per row)

Rank r: `r4dx_argmax_val_f32` over its `[vocab_local_]` row -> `(i_r, v_r)`, lowest local index among
equal maxima; `g_r = i_r + r * 124160`. D2H 8 B, `HostAllGather` 8 B. `MergeArgmax`: winner = rank 1
iff `v_1 > v_0`, else rank 0. Because rank 0's ids are all below rank 1's, a tie resolves to the lower
global id -- exactly `r4dx_argmax_f32` / `kernels::Argmax` over the full row (lowest index wins; NaN
undefined as today). **Data per row: 8 B per rank.**

### 7.4 Sampled (per row) -- lossless

Rank r: `r4dx_topk_lse_f32_ws` over its shard at `inv_t` -> `RowSummary_r` (k = 64, raw vals, local
ids, `lse_r`); on the host add `r * 124160` to every id; `HostAllGather` the 536-B `RowSummary`
structs (same process, same layout). `MergeRowSummaries`:

1. Two-way merge of the two sorted lists under canonical order (value descending, tie -> lower id),
   keep the first 64 -> `ids/vals`; `k = 64`.
2. `lse = float(LogAddExp(lse_0, lse_1))` in double: `m = max(a,b)`; `m == -inf -> -inf`; else
   `m + log(exp(a-m) + exp(b-m))`. Always called with inputs in rank order, so both ranks compute the
   same bits.
3. `vocab = 248320`, `inv_temperature` unchanged.

Then the unchanged `SampleFromSummary(merged, params, u)` with the one draw `u`.

Proof it stays exact: every member of the global top-64 is within the top-64 of its own shard, so the
merged list IS the global top-64 with the identical float values; every exact case of
`SampleFromSummary` reads only those values. The only approximate input, `lse`, has error <= max over
shards of the kernel's error (<= 1e-4 absolute, measured 4.8e-6) plus up to ~1 fp32 ulp of |lse|
(the per-shard and merged fp32 roundings; the double combination itself adds ~1e-16) -- 2.4e-4 at
|lse| in [2048, 4096), where T = 0.01 puts this model's rows -- inside the unchanged
`kRowSummaryLseRelTol = 1e-3` band, whose "exact or silent" guarantee is what the proof in
docs/sampling.md section 4 rests on. The `kMinSummaryTemperature` screen (`model.cpp:1006`) applies
unchanged; that extra ulp is TP's real cost, so TP gives no reason to lower it.

**Unresolved row** (`resolved == false`): each rank D2Hs its shard row (496,640 B), `HostAllGather`,
the concatenation in rank order is the full row in global id order; `SampleCanonical(full, 248320,
params, u)` with the **same** `u`. **Data per row: 536 B per rank; +496,640 B per rank on fallback.**

### 7.5 Full logits (`Prefill`, `DecodeStep`, `logits_out`, `ReadVerifyLogitsRow`)

`Model::GatherVocabRow(const float* shard_row_dev, float* full_host)`: D2H the shard, `HostAllGather`,
done. Both ranks hold identical `[248320]` rows (the facade `memcmp`s them). All four callers are
production paths, including `VerifyWindow`'s `logits_out` (sampled speculative rounds at
`0 < T < 0.01`, H4); `test_tp_emulation` covers that case explicitly at `T = 0.005` for both
`DecodeStepMtpSampled` and `DecodeStepDflashSampled` (10.1).

### 7.6 Speculative verify under a split vocab

`VerifyWindow(T)` launches T `r4dx_argmax_val_f32` (and, sampled, `LaunchRowSummaries` in <= 8-row
calls) into rank buffers, syncs, D2Hs, then ONE `HostAllGather` of `{T pairs, T summaries}`; merges
per row. `preds` and `round_summaries_` are then global and identical on both ranks, so the unchanged
`VerifyAndResolveRound` walk (one draw per emitted token, stop at the first mismatch) runs identically
on both ranks, and so do `CommitVerifiedWindow`, MTP reseed and DFlash inject. The greedy rule
("accept draft i iff merged argmax of row i == d_{i+1}") and sample-and-match are therefore unchanged
in meaning. **Data per round: T x 8 B greedy; T x (8 + 536) B sampled; + 496,640 B per unresolved row.**

---

## 8. MTP, DFlash2, vision, Reset / prefix cache, mrope, long context

### 8.1 MTP (`src/model/mtp_head.{h,cpp}`)

- Sharded like a body attention layer + MLP (4.2): `MtpHead(cfg_rank, w, max_draft, max_ctx,
  logits_rows, core::TpComm* comm = nullptr)` with `logits_rows = max(lm_head.N,
  draft_lm_head.N)` (TP=1: `vocab_size`, as today, since `draft_lm_head.N <= vocab_size`);
  `logits_dev_` is sized `logits_rows` (4.4). `attn_layer_` is built with `MakeAttnConfig(cfg, comm)`;
  `Draft` builds `Mlp(cfg, ..., comm)`; `prime_attn_layer_` is built with `MakeAttnConfig(cfg,
  nullptr)` and used only by `PrimeKv` (6.2). Its KV cache has 2 heads per rank. Every draft step
  runs the A2/A3 all-reduces on both ranks; `PrimeKv` runs none.
- `mtp.fc` replicated; `h_seed` (`mtp_seed_hidden_`) is the replicated residual.
- `Draft` must separate the two uses of today's `vocab` parameter: the embedding table stays global
  (`cfg.vocab_size`); the full-head argmax runs over `lm_head.N` (`logits_dev_` sized `lm_head_rows`).
- Full-vocab draft head under TP: the device-resident chain (argmax -> embedding gather with no host
  hop) is replaced, per step, by `r4dx_argmax_val_f32` -> `stream.Synchronize()` -> D2H 8 B -> H6
  `HostAllGather` -> `MergeArgmax` -> `seed_token_dev_.CopyFromHost(&tok, 1)` -> next step. Cost ~30-50
  us per draft step (~0.1-0.15 ms per K=3 round). TP=1 keeps today's chain.
- Reduced-vocab draft head: replicated, so both ranks compute the same subset argmax and
  `r4dx_gather_i32` mapping on device; the chain stays device-resident, no merge.
- Verification is `VerifyWindow` (7.6); losslessness argument unchanged.

### 8.2 DFlash2 (`src/model/dflash_draft.{h,cpp}`)

- Every rank loads the full drafter (`DflashDraft::Load`), with `DflashDraftOptions` extended:

  ```cpp
  int64_t lm_head_vocab = 0;     // rank shard rows (124160) -- sizes logits_dev_ and topk16
  int64_t vocab_offset = 0;      // rank * 124160
  int64_t global_vocab = 0;      // 248320; 0 => lm_head_vocab (TP=1)
  core::TpComm* comm = nullptr;
  std::shared_ptr<const DflashHostCodebooks> shared_codebooks;  // nullptr => read from the file
  ```

  The `mask_token_id` / `anchor_id` range checks (`dflash_draft.cpp:202, 525`) and the codebook size
  check (214) use `global_vocab`.
- `InjectFeatures`: unchanged; both ranks inject the same (replicated) captured features, so both
  rings stay identical.
- `DraftRound`: identical up to `lm_head(...)` (the target's split head via
  `MakeTargetLmHeadProvider`, N = 124160) and `r4dx_topk16_f32(..., lm_head_vocab_)` (no global
  scratch; safe per device). After the one D2H + sync: if `comm_`, add `vocab_offset` to the 8 x 16
  local ids, H7 `HostAllGather` of `{cand, unary}` (1 KiB), `MergeTop16` per row (top 16 of 32 under
  the same total order `r4dx_topk16_f32` uses) -> the merged `cand/unary` are exactly the full-vocab
  top-16. `gate` comes from the replicated drafter output and is identical. `SelectorWalk` then runs
  identically on both ranks, so the drafts, and hence the verify window T, agree.
- `Model::DecodeStepDflashImpl` is unchanged: draft (both ranks) -> `VerifyAndResolveRound` (7.6) ->
  inject accepted rows -> `CommitVerifiedWindow`.
- Why replicate instead of shard: the drafter is launch-bound (c = 3.26 ms of its 5.77 ms device
  time, `docs/dflash2.md` 6c); sharding saves at most ~1 ms of byte time and adds ~12 all-reduces
  per round plus a second sharding scheme (32/8 heads, grouped conv over `hidden`). Replication costs
  2.03 GiB on rank 1 and keeps the round symmetric with no extra transport.

### 8.3 Vision

- Rank 0 holds the tower (`vision_weights_on_this_rank = true` on rank 0 only). Rank 1 parses the
  config. `Model::HasVision()` (encode capability) is true on rank 0 only; new
  `Model::VisionSpliceEnabled()` = `container_.HasVisionConfig()` is true on both.
- Every `container_.HasVision()` use in `Model` is classified (grep it; at `aa54c20` there are
  exactly these):

  | `model.cpp` line | Today | Under TP (`tp.world > 1`) |
  |---|---|---|
  | 131-135 | `--vision on` and no vision tensors -> throw | test `container_.HasVisionConfig()` instead, so rank 1 (config parsed, weights not uploaded) does not throw |
  | 136-142 | construct `vision_` and log | unchanged: only the rank that uploaded the tower (rank 0) constructs it |
  | 143-149 | `kAuto` + tensors present + tower not loaded -> warning | suppressed when `!opts.tp.vision_weights_on_this_rank` (the tower lives on another rank by design) |
  | 355-363 | VRAM log line for the tower | unchanged (prints on rank 0 only) |
  | 379 (`EncodeImages`, via `Model::HasVision()`) | requires the tower | unchanged (only ever called on rank 0) |
  | 866-870 | image spans but no tower -> throw | `VisionSpliceEnabled()` |
  | 872-873 | `merge_size` from `container_.Vision().config` | from `container_.VisionCfg()` |
- `TpModel::EncodeImages`: solo command on rank 0; the merged rows go to `ImageRows::host`
  (pageable std::vector). `ImageSpan::embeds_on_host = true`.
- `Model::SpliceImageEmbeddings` (`model.cpp:500-503`): copy kind `sp.embeds_on_host ?
  hipMemcpyHostToDevice : hipMemcpyDeviceToDevice`. Both ranks splice identical rows. Cost ~2-4 ms per
  1024-token image per rank, against ~160 ms of encode.
- The CLI's `ImageBatch::embeds` and the server's `image_embeds_owned` become `ImageRows`.

### 8.4 `Model::Reset` and the server prefix cache

`TextModel::Reset()` = (recovery if needed) + `RunAll(m.Reset(); m.SetDflashInjectionEnabled(
dflash_injection_))`. `Model::Reset` has no collectives; it re-zeroes GDN state and resets host
counters, the DFlash ring counters and the mrope state on each rank.

**Server recovery after an error needs two fixes, one of them a bug on `main` today.**

1. `SetDflashInjectionEnabled` must not throw in `kNeedsRecovery` (2.4): `Engine::RunRequest` calls it
   (`engine.cpp:367-369`) before the prefix decision.
2. **`PrefixState::Invalidate()` does not force a reset today.** The catch block
   (`engine.cpp:980-987`) calls `prefix_.Invalidate()`, which clears `fed_`
   (`prefix_state.h:98-101`, identical to `Clear()`). With `fed_` empty, the next request's
   `Extend(full_tokens)` passes every check (`full_tokens.size() > 0`, `std::equal` over an empty
   range is true) and returns **all** of `full_tokens` as the "tail" (`prefix_state.h:60-73`). The
   engine then skips `model_->Reset()` (`engine.cpp:383-386`) and prefills the new prompt on top of
   whatever the failed request left in the KV/GDN state (`Model::Prefill` appends at `pos_`,
   `model.cpp:959-983`). At TP=1 that is silently wrong output after any mid-request exception. At
   TP=2 it is permanent failure: `Reset()` is never called, so every forward call throws
   `TpStateError`, the catch invalidates again, and the server answers 500 until restarted.

   Fix (in `src/server/prefix_state.h`): add `bool needs_reset_ = false;`. `Invalidate()` clears
   `fed_`/`fed_images_` **and** sets `needs_reset_ = true`; `Extend()` returns `std::nullopt` while
   `needs_reset_` is set; `Clear()` (called right after `model_->Reset()`, `engine.cpp:391`) clears
   it. The first request after construction is unaffected (`needs_reset_` starts `false`, the
   model is fresh). `tests/server/test_prefix_state.cpp` gains
   `TestInvalidateForcesResetPath` (`Commit`, `Invalidate`, then `Extend` of a longer prompt
   returns `nullopt`; after `Clear()` it extends again). It changes behaviour only on the error
   path (G2 rows never throw), but it IS a behaviour change, so on `tp2` it lands in P5 together
   with the server's TP support. Landing it on `main` now, independently of TP, is recommended and
   is the user's decision (Appendix C).

With both fixes, the request after any TP error runs: cached toggle -> `Extend` refuses ->
`Reset()` (recovery 2.5 + `Model::Reset` + toggle) -> normal request. G11's `-TpFault` case proves
it end to end on the production `--dflash` configuration.

### 8.5 mrope deltas

`mrope_active_`, `mrope_delta_`, `mrope_block_*` and `DflashDraft::rope_delta_` are host state computed
from identical inputs on both ranks; `RopePositionsForChunk` uploads the same rows on both. No change
besides the vision splice above.

### 8.6 Long context

`--max-ctx` applies per rank with half the KV heads, so the 262144 default costs 4.0 GiB of KV per
rank instead of 8.0. Decode gains grow with context (~1.6-1.7x at 131k-262k: the KV read splits).
Prefill gains shrink (~1.15x at 131k): prefill attention runs `(ceil(T/64), kv_heads_local, 1)`
workgroups, i.e. 2 per 64-token chunk per rank, and does not split in time. A split-KV prefill
attention is out of scope here (docs/perf.md already flags it as a single-GPU lever).

---

## 9. CLI / server flags, device selection, HIP_VISIBLE_DEVICES

### 9.1 Flags (identical on `r4dx-cli` -- `src/cli/cli_args.h` -- and `r4dx-server` -- `src/server/server_args.h`)

| Flag | Default | Meaning |
|---|---|---|
| `--tp N` | 1 | 1 = today's single-device engine (no other `--tp-*` flag allowed); 2 = tensor parallel |
| `--tp-mode real\|emulate\|noop` | real | emulate: both ranks on one device; noop: one rank, no-op all-reduce (timing only) |
| `--tp-devices a[,b]` | auto | process-visible HIP ordinals, rank r -> entry r (after `HIP_VISIBLE_DEVICES` remapping) |
| `--tp-rank r` | 0 | noop only: which shard to load |
| `--tp-ar-timeout-ms N` | 500 | all-reduce spin timeout, [10, 1500] |
| `--tp-ar-nb N`, `--tp-ar-nb-large N` | 4, 4 | expert: blocks per all-reduce for channel 0 / 1, [1, 64] |

Usage errors (`CliUsageError` / `ServerUsageError`): any `--tp-*` with `--tp 1`; `--tp` not in {1,2};
`--tp 2` with `--profile` or `--profile-prefill` (permanent in v1); `--tp-rank` outside noop.
**Staged rejections** (removed by the phase that implements them): the `--tp*` flags appear in P2b;
until P4 `--tp-mode real`; until P5 `--tp 2` with `--mtp > 0`, `--dflash`, `--vision on`,
`--image`/`/image`, and `r4dx-server --tp 2` altogether. The same rejections are enforced a second
time in `TpModel::Load` (2.9 step 1), so tests and tools that bypass the arg parsers hit them too.

`R4DX_TP_FAULT=<rank>:<n>:<kind>` (environment, test-only, read by `TpModel::Load`; see
`TpOptions`) arms one fault injection in the production binaries; `tools/server/smoke.ps1
-TpFault` uses it. It is ignored at `--tp 1`.

`--stats` with `--tp 2` prints one VRAM line per rank and
`[stats] tp: mode=real devices=1,0 ar_calls=<ch0>/<ch1> host_exchanges=<n> max_exchange_wait=<us>`.

### 9.2 Device selection and `HIP_VISIBLE_DEVICES`

- **TP=1**: unchanged. No `hipSetDevice`; the device is whatever `HIP_VISIBLE_DEVICES` exposes
  (project rule: `1`).
- **`--tp-devices auto`** = descending ordinals starting at the last visible one: real mode with 2
  visible devices -> `1,0` (rank 0 = ordinal 1, rank 1 = ordinal 0); emulate/noop -> the last visible
  ordinal only. Consequences with `HIP_VISIBLE_DEVICES` **unset** (the recommended TP setting): rank 0
  = physical device 1 (headless -- it also runs the vision encode), rank 1 = physical device 0
  (desktop). With `HIP_VISIBLE_DEVICES=1`, emulate/noop pick ordinal 0 = physical 1, so every
  single-device TP run respects the device-1 rule automatically.
- Real mode with fewer than 2 visible devices fails at load:
  `--tp 2 --tp-mode real needs two visible HIP devices; HIP_VISIBLE_DEVICES=<value> exposes <n>.
  Unset it (or set it to 0,1).`
- Real mode refuses two ordinals with the same PCI bus id, or differing `gcnArchName`.
- The load log prints ordinal -> PCI bus id -> name per rank, so a remapped `HIP_VISIBLE_DEVICES`
  (e.g. `1,0`) is visible in the log.
- The user has authorized TP work on both GPUs. Single-device TP work (emulate/noop, P2a/P2b) stays
  on device 1, except the one recorded G3 cross-check of rank 1's step on device 0 (P2a). Real-mode
  runs (P3 2-GPU tests, P4, P5) use both.
- **Every full-v6 run on device 1 needs the production server stopped** -- it holds ~28 GiB of
  the card's 31.86 GiB. That includes the P2a gates (G2's v6 rows ~16-19 GiB with `--dflash`, G3's
  noop shard ~10 GiB plus the TP=1 reference ~16 GiB in separate processes), G4's emulated run (two
  shards + two embedding mirrors on one card, ~19 GiB) and everything in P3-P5. The 4-layer
  `l4-*` test containers fit next to it. `tools/tp/tp1_identity.ps1`, `tool_tp_step_bench`,
  `tool_tp_soak` and `run_tests.ps1 -TwoGpu` run a **pre-flight**: `Get-Process r4dx-server
  -ErrorAction SilentlyContinue` must be empty (scripts), and the C++ tools check `hipMemGetInfo`
  free bytes on every device they use against their own requirement and exit 1 with `need <x> GiB
  free on HIP device <d>, have <y> GiB -- is the production server running?`.

---

## 10. Testing

### 10.1 Test inventory

| Test | Kind | Registered | Device(s) | Covers | Phase |
|---|---|---|---|---|---|
| `tests/model/test_tp_config.cpp` | CPU | ctest (plain `add_test`) | none | `ModelConfig::Shard`: every field of 3.1 for the real `config.json` values and a synthetic config; every divisibility failure throws naming the field; `VocabShard*` | P1 |
| `tests/model/test_tp_shard.cpp` | CPU | ctest | none | `RuleFor` for every tensor family in 4.2 (incl. `mtp.*`, bare/`.bf16.w`, unknown name throws); byte-exact `Gather(Plan*(pack(W))) == pack(slice(W))` (10.2) | P1 |
| `tests/model/test_tp_vocab_merge.cpp` | CPU | ctest | none | `MergeArgmax` (ties across the shard boundary, -inf rows); `MergeRowSummaries` vs a full-row CPU summary (ids/vals exact, lse within 1e-6) on random, tie-heavy, peaked, flat rows, V in {4096, 248320}; `SampleFromSummary(merged)` == `SampleCanonical(full)` for 100000 u x the six filter configs of `test_summary_sampler.cpp` whenever resolved; `MergeTop16` vs full top-16 with ties | P1 |
| `tests/core/test_tp_host_exchange.cpp` | CPU (threads) | ctest | none | 2 threads x 1,000,000 all-gathers of varying sizes, contents verified; Abort wakes a waiter with `TpAbortedError`; timeout -> `TpTimeoutError`; `Reset` zeroes `arrive[]` and `gen_[]` (an asymmetric abort -- rank 0 aborted before its increment, rank 1 after -- followed by `Reset` gives correct contents on the next 1,000 gathers) | P1 |
| `tests/model/test_tp_rank_worker.cpp` | CPU (threads) | ctest | none | `RankWorker` + `CompletionGroup` (2.2): 2 workers x **1,000,000 empty commands with `WorkerTiming{0 us, 0 us}`** (every hand-off goes through both condvars) and no hang (the test's own 120 s alarm); 10,000 commands with random 0-2 ms sleeps on one side; an exception in a command comes back through `TakeError()`; the progress-watchdog helper fires only when no heartbeat moves (fake clock); destructor with a busy worker waits for it and never overwrites `cmd_` | P1 |
| `tests/model/test_tp_loader.cpp` | GPU | ctest, `HIP_VISIBLE_DEVICES=1`, SKIP 77 | dev 1 | load `qwen38-27b-l4-allmtp.r4dx` at `{world 2, rank r}` for r = 0,1 in each of the 4 layouts: every uploaded buffer D2H == `Gather(Plan(...))` of the file bytes; every `QuantLinear::N/K` == rank shape; bf16 layout: rank 0 ∪ rank 1 slices reassemble the full tensors exactly; `EmbedTokensHost()` of both ranks is the same pointer when shared | P2a |
| `tests/model/test_tp_emulation.cpp` | GPU | ctest, dev 1, SKIP 77 | dev 1 | TP=1 `Model` vs `TpModel(emulate)` on `l4-allmtp`, all 4 layouts: prefill 40 and 70 tokens (1 and 2 chunks) + 16 teacher-forced `DecodeStep` rows, per-row rel L2 of logits <= 1e-2 (bf16, w4a16) / <= 5e-2 (w4a8, mxfp4); `DecodeStepGreedy` == argmax of `DecodeStep` path (exact); `DecodeStepSampled` vs `DecodeStep`+`SampleCanonical` on the same TpModel, 3 configs x 3 seeds, trajectories **exactly** equal (validates 7.4 end to end); one merged summary vs `r4dx_topk_lse_f32` over the gathered row (ids/vals exact, lse <= 1e-4); `Reset()` then rerun == first run byte for byte; fault injection (rank 1 throws at AR #37): the injected exception reaches the caller, the next forward call throws `TpStateError`, `SetDflashInjectionEnabled` and every cached accessor still work in `kNeedsRecovery`, `Reset()` recovers, both endpoints report equal `CallCounts()`, rerun == fresh run byte for byte; the same with an asymmetric fault (kind 1, stall -> peer timeout). Every case ends with `g_tp_collective_allocs == 0` (2.7) | P2b |
| `test_tp_emulation.cpp`, P5 additions | GPU | same | dev 1 | **MTP** K=3 greedy/sampled (bookkeeping lockstep: committed positions == emitted tokens each round; sampled rounds vs plain sampled decode via `tests/model/sampled_equality.hpp`'s classifier) and the reduced-vocab head on `l4-mtp-draftvocab`. **DFlash** k=7 on `l4-allmtp` + `ProductionDrafterPath()` (`test_container_path.h:131`), greedy and seeded sampled, 3 prompts: (a) **H7 exactness** -- a test hook (`Model::DflashDebugLastTop16()`, compiled only with `R4DX_TP_TESTING`) returns the merged `cand/unary` of the last round; the test gathers the drafter's full `[8, 248320]` logits through `GatherVocabRow`, computes the CPU top-16 with `r4dx_topk16_f32`'s total order, and requires exact equality; (b) **H6 exactness** for MTP full-vocab drafts the same way; (c) **lockstep bookkeeping** -- after every round `DflashInjectedCount() == PositionCount()` on both ranks and the drafts/`walk_len` agree (the facade already compares them); (d) **lossless-by-construction** -- every emitted greedy token equals the merged argmax of its verify row, recomputed from a gathered full row. **Tiny temperature**: `DecodeStepMtpSampled` and `DecodeStepDflashSampled` at `T = 0.005` (the `logits_out` path, H4) equal `SampleCanonical` over gathered rows | P5 |
| `tests/kernels/test_tp_allreduce_cpu_peer.cpp` | GPU | ctest, dev 1 | dev 1 | the engine AR kernel against a **CPU thread** acting as rank 1 through a real pinned mailbox: 100,000 ARs at {10240, 81920, 174080, 655360} B, bit-exact; timeout path (CPU peer silent) returns within timeout +10%, ABORT word set, next call skips (sticky); seq base `0xFFFFFF00` crosses the 2^32 wrap cleanly | P3 |
| `tests/kernels/test_tp_allreduce_2gpu.cpp` | GPU | ctest, LABEL `tp2gpu`, **opt-in** (below) | dev 0+1 | `HostMailboxComm` end to end: 1,000,000 ARs, mixed sizes on both channels, interleaved with VRAM filler kernels that dirty L2, verified bit-exact every batch (tp_bench's hash pattern); abort propagation (rank 1 `Abort()` -> rank 0 kernel exits < 5 ms, both see `TpAbortedError`); `Recover()` then 10,000 more clean ARs | P3 |
| `tests/model/test_tp_real_vs_emulation.cpp` | GPU | ctest, LABEL `tp2gpu`, opt-in | dev 0+1 | on `l4-allmtp` w4a16 and mxfp4: a fixed script (prefill 70, 16 full-logit steps, 16 greedy, 16 seeded sampled) run in emulate mode then real mode -> **byte-identical** outputs; P5 extends with MTP K=3 and DFlash k=7 (`ProductionDrafterPath()`) greedy and seeded-sampled rounds, byte-identical tokens, `walk_len` and final logits | P4, P5 |
| `tests/model/tool_tp_step_bench.cpp` | GPU tool | built, not `add_test`'d | dev 1 (G3), dev 0 (recorded cross-check) | per-rank decode ms/token with `NoopComm` and a bare `Model` (no `TpModel` needed), and the TP=1 baseline with `--tp1`, same prompt/protocol; prints both and the ratio; pre-flight VRAM check (9.2) | P2a |
| `tests/kernels/tool_tp_ar_stress.cpp` | GPU tool | built only | dev 0+1 | `--count 10000000 --pattern isolated|decode` through `HostMailboxComm` (G5) | P3 |
| `tests/kernels/tool_tp_ar_latency.cpp` | GPU tool | built only | dev 0+1 | tp_bench's decode pattern using the engine comm, with all three interleaved conditions -- (a) AR, (b) stand-in kernel on the same grid, (c) fillers only -- 128 ARs per token, 60 MiB filler, 4 MiB dirty; reports **`L_vs_no_ar_kernel = (T_a - T_c)/128`** (the G5 number) and `L_vs_standin = (T_a - T_b)/128` at 10 KiB, 80 KiB, 640 KiB, for nb in {4, 8, 16} on channel 1 | P3 |
| `tests/model/tool_tp_soak.cpp` | GPU tool | built only | dev 0+1 | G8 (10.5) | P4 |
| `tests/model/tool_teacher_forced_logprobs.cpp` | GPU tool (existing) | built only | any | gains `--tp 2 --tp-mode real|emulate|noop --tp-devices` and `--embed-device-resident`; `teacher_forced.h::RunSegment` takes `TextModel&` | P2b |
| `tests/server/test_prefix_state.cpp` (existing) | CPU | ctest | none | + `TestInvalidateForcesResetPath` (8.4) | P5 |
| `tools/tp/tp1_identity.ps1` | script | -- | dev 1 | G2 (10.3) | P2a |

**Two-GPU tests are opt-in, not just labelled.** A per-test `HIP_VISIBLE_DEVICES` property would
override the `win-hip` test preset's `HIP_VISIBLE_DEVICES=1` (`CMakePresets.json:53-55`), so a
plain `ctest --preset win-hip` would run them on the desktop GPU. Therefore:

- `tp2gpu` tests set **no** `HIP_VISIBLE_DEVICES` property. `main()` first checks the environment
  variable `R4DX_TP2GPU`; unless it is exactly `1`, it prints `SKIP: two-GPU test; run
  tests\run_tests.ps1 -TwoGpu` and exits 77. Then it requires >= 2 visible devices (else 77), then
  the per-device free-VRAM pre-flight (9.2; failing it is exit 1, not a skip).
- `tests/run_tests.ps1` (today: sets `HIP_VISIBLE_DEVICES=1`, runs ctest with no label filter,
  `run_tests.ps1:39,46`): the default path is unchanged except it adds `-LE tp2gpu` and makes sure
  `R4DX_TP2GPU` is not set. New `-TwoGpu` switch: `Get-Process r4dx-server -ErrorAction
  SilentlyContinue` must be empty (else throw "stop the production server first"); remove
  `HIP_VISIBLE_DEVICES` from the process environment; set `R4DX_TP2GPU=1`; run
  **`ctest --test-dir build\win-hip -L tp2gpu --output-on-failure`** -- deliberately *not*
  `--preset win-hip`, because the preset would inject `HIP_VISIBLE_DEVICES=1` into every test;
  restore both variables in a `finally`.
- Net effect: under any `ctest --preset win-hip` (the P1 gate, developer runs) a `tp2gpu` test sees
  one device and no opt-in, and skips twice over; only `run_tests.ps1 -TwoGpu` ever puts a test on
  device 0.

### 10.2 CPU slicer tests (`test_tp_shard.cpp`) -- exact contents

For each layout, using the converter's own header-only quantizers/packers
(`QuantizeInt4Asymmetric` + `PackW4Nibbles` + `PackW4A16Scales` at g = 64 and 128;
`QuantizeInt4SymmetricPinned8` + `PackW4A8Scales`; `QuantizeInt4AsymmetricSearch` with a random
importance vector; `QuantizeMxfp4` and `QuantizeMxfp4Search` + `PackMxfp4Wq` + `PackMxfp4Ws`;
`EncodeBf16`) on random `W` with outliers:

- **rows**, single segment: N = 256, K = 1024, rank rows [0,128) / [128,256);
- **rows, fused**: a qkv-shaped `N = 320` = segments {64, 64, 192} -> rank 0 {0-32, 64-96, 128-224};
  gate_up-shaped {96, 96}; qg-shaped with 4 "heads" of 32 rows;
- **cols**: N = 64, K = 1536 (3 groups of 512) -> [0,768) / [768,1536); K = 1088 (= 17 x 64) at g=64
  (mirrors 17408 / 8704 legality);
- assertion: `Gather(full, PlanRows/PlanCols(...)) == pack(W[rows or cols])` **byte for byte** for
  every part, **except** mxfp4 K-slice `wref`, where the assertion is `== full.wref` and, separately,
  a CPU dequant of `(sliced wq, sliced ws, full wref)` equals the matching columns of the dequant of
  the full packed tensor element for element;
- misalignment (row range not % 16, col range not % group) throws;
- `w4a8.ws` stride is 4 B per (t, g, r) (a uint16 reading fails the test by construction).

### 10.3 TP=1 byte-identity guard (`tools/tp/tp1_identity.ps1`)

Baseline: before the first `src/` change, with the tree at `aa54c20`: `.\build.ps1`, then copy
`build\win-hip\src\cli\r4dx-cli.exe`, `build\win-hip\src\server\r4dx-server.exe`,
`build\win-hip\tests\model\tool_teacher_forced_logprobs.exe` to `build\baseline\` (never rebuilt).
The script first runs the pre-flight of 9.2 (production server stopped), then runs each row with
the baseline and the candidate (`HIP_VISIBLE_DEVICES=1`, no `--tp`) and compares SHA-256 of stdout
generated text and of `--dump-token-ids` files, plus the timing-free `[stats] mtp/dflash/sampled`
stderr lines (N26). The baseline stays valid for the whole branch
because no phase changes a TP=1 byte: the TP=2 tuning rows live in their own table (2.7), and the
`PrefixState` fix only touches the server's error path.

| Row | Command (v6 unless noted) |
|---|---|
| 1 | standard protocol, plain greedy |
| 2 | standard protocol + `--dflash D:\models\r4dx\qwen38-27b-dflash2-w4a16-g64.r4dx --dflash-k 7` |
| 3 | standard protocol + `--mtp 3` |
| 4 | `--temperature 0.7 --top-k 20 --top-p 0.8 --seed 1` plain |
| 5 | row 4 + `--dflash ... --dflash-k 7` |
| 6 | `tool_teacher_forced_logprobs` on `qwen38-27b-l4-allmtp.r4dx`, layouts bf16/w4a16/w4a8/mxfp4, `--layers 4`, kl_corpus tokens -> SHA of every `*.logprobs.f16` |
| 7 | `--vision on --image tools\reference\golden_out\vision_test_image.png --prompt "What is in this picture?"` plain greedy (the golden image `tests/vision/test_preprocess.cpp` decodes; if the gitignored `golden_out` is absent, the script SKIPs this row and says so) |
| 8 | row 4 + `--mtp 3` (added in P2a review, N26) |
| 9 | `--chat`, two user turns through stdin, greedy (added in P2a review, N26) |

Gate: all equal. First run at the end of P2a; re-run at the end of P3, P2b, P4, P5.

### 10.4 Emulation vs real, KL vs TP=1

```powershell
$py = "<reference venv>\Scripts\python.exe"
$tf = "build\win-hip\tests\model\tool_teacher_forced_logprobs.exe"
$m  = "D:/models/r4dx/qwen38-27b-v6.r4dx"
$tok = "tools\reference\kl_corpus\tokens.json"
# TP=1 (device 1)
$env:HIP_VISIBLE_DEVICES='1'
& $tf --model $m --layout w4a16 --tokens $tok --out-dir tools\reference\kl_out\v6_tp1 --max-ctx 4096 --vision off
# TP=2 emulated (device 1)  [P2b]
& $tf --model $m --layout w4a16 --tokens $tok --out-dir tools\reference\kl_out\v6_tp2emu --max-ctx 4096 --vision off --tp 2 --tp-mode emulate
& $py tools\reference\kl_report.py --ref-dir tools\reference\kl_out\ref --test-dir tools\reference\kl_out\v6_tp2emu --tokens $tok --out tools\reference\kl_out\kl_v6_tp2emu.json
& $py tools\reference\kl_report.py --ref-dir tools\reference\kl_out\v6_tp1 --test-dir tools\reference\kl_out\v6_tp2emu --tokens $tok --out tools\reference\kl_out\kl_tp1_vs_tp2emu.json
# G6 [P4]: regenerate the emulation reference WITH THE SAME BINARY, then the real run
$env:HIP_VISIBLE_DEVICES='1'
& $tf --model $m --layout w4a16 --tokens $tok --out-dir tools\reference\kl_out\v6_tp2emu_g6 --max-ctx 4096 --vision off --tp 2 --tp-mode emulate
Remove-Item env:HIP_VISIBLE_DEVICES
& $tf --model $m --layout w4a16 --tokens $tok --out-dir tools\reference\kl_out\v6_tp2real --max-ctx 4096 --vision off --tp 2
(Get-FileHash $tf).Hash | Out-File -Encoding utf8 tools\reference\kl_out\v6_tp2real\binary.sha256
Get-ChildItem tools\reference\kl_out\v6_tp2real\*.logprobs.f16 | ForEach-Object {
  $e = Join-Path tools\reference\kl_out\v6_tp2emu_g6 $_.Name
  if ((Get-FileHash $_).Hash -ne (Get-FileHash $e).Hash) { throw "G6 FAIL $($_.Name)" } }
```

G4: `kl_v6_tp2emu.json` overall mean KL <= 0.0435, top-1 >= 90.43%; `kl_tp1_vs_tp2emu.json` is
reported (expected mean KL ~1e-3). G6: no throw; the emulate and real dumps come from one binary
(hash recorded), so a P3/P4 change to kernels, tunings or defaults can never make G6 compare
against a stale reference.

### 10.5 Soak (`tool_tp_soak`)

`tool_tp_soak.exe --model v6 --layout w4a16 --minutes 60 --max-ctx 8192 [--dflash <g64 drafter>]
--json build\logs\tp_soak.json`: one `TpModel` (real) for the whole run; loop: pick a random slice
(16-2048 tokens) of a random kl_corpus segment, `Reset()`, `Prefill`, then 32-512 decode tokens in a
random mode (greedy / sampled seeded / DFlash greedy / DFlash sampled when loaded); every 10th
iteration re-run a fixed 512-token canary prompt greedily for 128 tokens and require the output to
equal the first canary run exactly. Record iterations, tokens, `TpCommStats`, max exchange wait,
per-rank VRAM at start and end. Pass: exit 0, 0 aborts, 0 divergences, canary always equal, per-rank
VRAM drift <= 64 MiB, `g_tp_collective_allocs == 0` (2.7), and no TDR since the start: no
Application-log Windows Error Reporting event for LiveKernelEvent 141 and no System event 4101
(on this box a TDR shows up only as the former, N44).

---

## 11. Phases

Every phase builds with `.\build.ps1`, passes `.\tests\run_tests.ps1`, and leaves `--tp`-less
behaviour byte-identical (G2 from P2a on). Commit per phase on `tp2`; nothing merges to `main` until
P4's gates are green.

**Order: P1 -> P2a -> P3 -> P2b -> P4 -> P5.** P2a answers "is the per-rank step fast enough" (G3)
with ~1,650 LOC; P3 answers "is the in-engine all-reduce as fast as tp_bench" (G5). P3 depends only
on P1's interfaces (plus the `r4dx_tp` CMake target, which P2a creates -- if P3 is ever done first,
it creates the target). P2b -- the threading, emulation and `TpModel` integration, ~2,700 LOC --
starts only after **both** economic gates are green; if either misses, stop and bring the numbers
to the user (R3, R1) before writing it.

### P1 -- CPU only: config split, slicer, vocab merges, host exchange, rank worker (~2,400 LOC)

| File | Change | LOC |
|---|---|---|
| step 0 | freeze `build\baseline\` (10.3) at `aa54c20` **before** editing `src/` | -- |
| `src/model/model_config.h` | `tp_world`, `tp_rank`, `VocabShardSize/Begin`, `IsShard`, `Shard()` with validation (3.1) | +80 |
| `src/model/tp/tp_shard.h`, `tp_shard.cpp` (new, target `r4dx_tp_shard`, no HIP) | 5.2 | 400 |
| `src/model/tp/tp_vocab.h` (new, header-only) | 7.2 helpers | 160 |
| `src/model/tp/tp_rank_worker.h` (new, header-only, no HIP) | 2.2: `RankWorker`, `CompletionGroup`, `WorkerTiming`, progress-watchdog helper | 220 |
| `src/core/include/r4dx/core/tp_host_exchange.hpp` (new) | 6.6 (incl. `gen_[]`, `SetHeartbeat`) | 190 |
| `src/core/include/r4dx/core/tp_comm.hpp` (new) | 6.1 interface + errors (no implementations yet) | 100 |
| `src/model/CMakeLists.txt`, `tests/model/CMakeLists.txt`, `tests/core/CMakeLists.txt` | `add_library(r4dx_tp_shard STATIC tp/tp_shard.cpp)`; the five CPU tests as plain `add_test` (no HIP env) | +15 |
| tests: `test_tp_config.cpp` 150, `test_tp_shard.cpp` 450, `test_tp_vocab_merge.cpp` 280, `tests/core/test_tp_host_exchange.cpp` 180, `tests/model/test_tp_rank_worker.cpp` 200 | 10.1, 10.2 | 1,260 |

Gate (G1):
```powershell
.\build.ps1
$env:HIP_VISIBLE_DEVICES='1'
ctest --preset win-hip -R "test_tp_(config|shard|vocab_merge|host_exchange|rank_worker)" --output-on-failure
.\tests\run_tests.ps1
git diff --stat aa54c20 -- src/   # only model_config.h (additive) and new files
```

### P2a -- sharded loader, NoopComm, hooks, vocab-split head, per-rank step on device 1 (~1,650 LOC)

Only what G3 and G2 need. No threads, no `TpModel`, no CLI flag: `tool_tp_step_bench` drives a bare
`Model` loaded as one shard with a `NoopComm`.

| File | Change | LOC |
|---|---|---|
| `src/model/container.h/.cpp` | `ContainerLoadOptions`, sharded path (5.1), `GlobalConfig`, `HasVisionConfig`/`VisionCfg`, shared embed, embed decision, vision config parse | +320 |
| `src/model/gdn_layer.h/.cpp`, `mlp.h/.cpp`, `attention/.../types.hpp`, `attention_layer.hpp`, `attn_config.h` | comm pointer + A1/A2/A3 (6.2) | +45 |
| `src/model/linear.h/.cpp` | `thread_local` cache; `SetTp2TuningForThisThread`; `tp2::` table lookup first on TP threads (2.7) | +45 |
| `src/model/gemm_tuning_table_tp2.inc` (new, generated) + `tools/profile/tune_gemm.py` (`SHAPES` entries only) | sweep below | ~70 + 10 |
| `src/kernels/src/r4dx_kernels.hip`, `kernels.h` | atomic launch counter; `r4dx_argmax_val_f32` (7.2) | +60 |
| `src/model/tp/tp_comm_noop.{h,cpp}` (new) + `src/model/CMakeLists.txt` (new target `r4dx_tp`, linked by `r4dx_model`) | 6.4 `MakeNoopComm` | 90 + 15 |
| `src/model/model.h/.cpp` | `ModelOptions::tp` (3.3), `comm_`, `vocab_local_/offset_`, `GlobalConfig()`, `SetTp2TuningForThisThread(true)` when `tp.world > 1`, `GatherVocabRow`, H1, H3 greedy + full-row, the plain-path lines of 7.2's table; under TP the sampled and speculative paths throw `TpUnsupportedError` until P2b, the profiled methods always | +230 |
| tests: `test_tp_loader.cpp` 320, `tool_tp_step_bench.cpp` 280; `tools/tp/tp1_identity.ps1` 160 | 10.1, 10.3 | 760 |

Tuning sweep (device 1, production server stopped, group-64 pyd per `tune_gemm.py`'s docstring).
The first run **creates** the TP table (no `--append`), the second appends to it; neither touches
`gemm_tuning_table.inc`:

```powershell
$env:HIP_VISIBLE_DEVICES='1'; $env:R4DX_LIBR4D_BUILD='C:\Users\pay20\dev\libr4d\build-win\g64'
& $py tools\profile\tune_gemm.py --layouts w4a16 --w4a16-group 64 --out src\model\gemm_tuning_table_tp2.inc `
  --shapes tp2.gdn.in_proj_qkv,tp2.gdn.in_proj_z,tp2.out_proj,tp2.mlp.gate_up,tp2.mlp.down,tp2.lm_head
& $py tools\profile\tune_gemm.py --layouts bf16 --append --out src\model\gemm_tuning_table_tp2.inc --shapes tp2.attn.kv
```
with `SHAPES` entries `("tp2.gdn.in_proj_qkv", 5120, 5120)`, `("tp2.gdn.in_proj_z", 3072, 5120)`,
`("tp2.out_proj", 5120, 3072)` (gdn.out_proj and attn.o), `("tp2.mlp.gate_up", 17408, 5120)`,
`("tp2.mlp.down", 5120, 8704)`, `("tp2.lm_head", 124160, 5120)`, `("tp2.attn.kv", 512, 5120)`.
(`attn.qg` at TP=2 is (6144, 5120) = TP=1's `gdn.in_proj_z`; the main table's rows serve it through
the fallback-to-main-table rule of 2.7.)

Gates (production server stopped):
```powershell
.\tests\run_tests.ps1                                              # incl. test_tp_loader
powershell -File tools\tp\tp1_identity.ps1 -Baseline build\baseline -Candidate build\win-hip   # G2
$env:HIP_VISIBLE_DEVICES='1'
$m  = "D:\models\r4dx\qwen38-27b-v6.r4dx"; $sb = "build\win-hip\tests\model\tool_tp_step_bench.exe"
& $sb --model $m --layout w4a16 --max-ctx 2048 --tp1 --tokens 128 --repeats 3 --json build\logs\tp_step_tp1.json
foreach ($r in 0,1) { & $sb --model $m --layout w4a16 --max-ctx 2048 --rank $r --tokens 128 --repeats 3 `
    --json build\logs\tp_step_r$r.json }
$t1 = (Get-Content build\logs\tp_step_tp1.json | ConvertFrom-Json).median_ms
foreach ($r in 0,1) {                                                                        # G3
  $x = (Get-Content build\logs\tp_step_r$r.json | ConvertFrom-Json).median_ms
  "rank $r : $x ms vs TP=1 $t1 ms -> ratio $($x / $t1)"; if ($x / $t1 -gt 0.633) { throw "G3 FAIL rank $r" } }
# recorded, not gated: rank 1's shard on device 0 (the desktop card it runs on in production)
$env:HIP_VISIBLE_DEVICES='0'
& $sb --model $m --layout w4a16 --max-ctx 2048 --rank 1 --tokens 128 --repeats 3 --json build\logs\tp_step_r1_dev0.json
```
`tool_tp_step_bench`: loads one shard with `MakeNoopComm(2, rank)` (or TP=1 with `--tp1`), renders
the standard prompt through the real chat template (`--think off`; links `r4dx_tokenizer`, like
`tool_vocab_calib`) and prefills it, then times 128 `DecodeStepGreedy` calls feeding a fixed token
cycle taken from the prompt (so garbage no-op output cannot change the work or hit EOS); median of
`--repeats`; prints ms/token and tok/s; `--json` writes `{median_ms, runs_ms[], rank, device_name,
pci_bus}`. Pass: G2 all equal; G3 both ratios <= 0.633. Record the device-0 number and its
difference from device 1 in the phase commit message (input to R2). If G3 fails: stop; see R3.

### P3 -- HostMailbox all-reduce in `src/kernels` + stress in ctest (~2,650 LOC)

Runs after P2a and before P2b (see the order note above).

| File | Change | LOC |
|---|---|---|
| `src/kernels/src/r4dx_tp_kernels.hip`, `tp_kernels.h` (new) | 6.3.1 port (flag kernel, Status, spin, publish, `r4dx_tp_add_bf16`, `r4dx_tp_clock_probe`) | 520 |
| `src/kernels/CMakeLists.txt`, `src/kernels/check_tp_isa.cmake` (new) | second hipcc object (WGP mode, r4dx_kernels' flags); ISA check (6.3.5) | 60 |
| `src/model/tp/tp_group.{h,cpp}` (new), `tp_comm_host_mailbox.cpp` (new) | 6.3.2-6.3.9, mailbox zero on creation, recovery (2.5), `HostU32Load/Store` | 720 |
| `tests/kernels/test_tp_allreduce_cpu_peer.cpp` | 10.1 | 380 |
| `tests/kernels/test_tp_allreduce_2gpu.cpp` | 10.1, label `tp2gpu`, `R4DX_TP2GPU` opt-in | 460 |
| `tests/kernels/tool_tp_ar_stress.cpp`, `tool_tp_ar_latency.cpp` | 10.1 (latency tool runs conditions a, b and c) | 440 |
| `tests/run_tests.ps1` (`-LE tp2gpu` default, new `-TwoGpu`), `tests/kernels/CMakeLists.txt` | 10.1 | 60 |

Gates (G5; production server stopped for the two-GPU lines):
```powershell
$env:HIP_VISIBLE_DEVICES='1'; .\tests\run_tests.ps1                  # includes the CPU-peer test; tp2gpu excluded
.\tests\run_tests.ps1 -TwoGpu                                         # 1M-AR stress + abort/recover (script handles env)
Remove-Item env:HIP_VISIBLE_DEVICES
build\win-hip\tests\kernels\tool_tp_ar_stress.exe --count 10000000 --pattern decode --json build\logs\tp_stress.json
build\win-hip\tests\kernels\tool_tp_ar_latency.exe --json build\logs\tp_latency.json
$env:HIP_VISIBLE_DEVICES='1'; powershell -File tools\tp\tp1_identity.ps1 -Baseline build\baseline -Candidate build\win-hip   # G2
```
Pass: 0 mismatches / 0 timeouts in 10M; in the decode pattern `L_vs_no_ar_kernel`(10 KiB) <= 9.0 us
and (80 KiB) <= 16.0 us (the vs-stand-in values are recorded next to them); record L(640 KiB) for nb
4/8/16 and set the `--tp-ar-nb-large` default to the best one (only if it is at least 10% better
than nb=4, and only after a 1M-AR stress at that nb); the ISA check passes; G2.

### P2b -- emulation, threading, `TpModel`, `TextModel`, CLI `--tp 2 --tp-mode emulate` (~2,700 LOC)

Starts after P3's G5 is green.

| File | Change | LOC |
|---|---|---|
| `src/model/gdn_state.h` | `GdnControlCache::Prewarm/Freeze` (2.7) | +40 |
| `src/kernels/src/r4dx_kernels.hip`, `kernels.h` | `r4dx_topk_lse_f32_ws` + `r4dx_topk_lse_workspace_bytes()` | +70 |
| `src/model/tp/tp_comm_emulated.cpp` (new), `tp_group.{h,cpp}` (emulate mode, recovery counters) | 6.5, 2.5 | 250 |
| `src/core/include/r4dx/core/tp_alloc_guard.hpp` (new), `device_buffer.hpp` | 2.7 allocation guard | +40 |
| `src/model/model.h/.cpp` | H2-H5, summary merges (7.4), `_ws` summaries under TP, `VerifyWindow` under TP (7.6), the rest of 7.2's table, `TpWarmup()`; lift the P2a `TpUnsupportedError`s | +240 |
| `src/model/model_types.h` (new), `model.h` aliases | `ImageSpan`, `ImageRows`, `StepProfile` (2.8) | 90 |
| `src/model/text_model.h`, `local_text_model.{h,cpp}` (new) | 2.8 | 300 |
| `src/model/tp_model.{h,cpp}` (new) | 2.2-2.9 for modes noop/emulate: `RankSlot`, `RunAll/RunOne/RunCollective`, state machine, recovery, warm-up, staged rejections | 800 |
| `src/vision/image_prompt.h` | `ImagePlaceholderSpan::embeds_on_host` (always `false` until P5) | +2 |
| `src/cli/cli_args.h`, `main.cpp` | `--tp*` flags (real/mtp/dflash/vision-on/image rejected), `unique_ptr<TextModel>`, `ImageRows`, per-rank VRAM, pre-load `VramUsedGiB()` only at `--tp 1` | +180 |
| `tests/model/teacher_forced.h`, `tool_teacher_forced_logprobs.cpp` | `TextModel&`, `--tp*`, `--embed-device-resident`, `NumLoadedLayers()`, `Vram()` | +70 |
| tests: `test_tp_emulation.cpp` | 10.1 (P2b row) | 600 |

Gates (production server stopped):
```powershell
.\tests\run_tests.ps1                                              # incl. test_tp_emulation
powershell -File tools\tp\tp1_identity.ps1 -Baseline build\baseline -Candidate build\win-hip   # G2
# G4: section 10.4 commands (TP=1 and emulate)
$env:HIP_VISIBLE_DEVICES='1'
build\win-hip\src\cli\r4dx-cli.exe --model D:\models\r4dx\qwen38-27b-v6.r4dx --layout w4a16 --vision off `
  --think off --temperature 0 --max-tokens 64 --max-ctx 2048 --tp 2 --tp-mode emulate `
  --prompt "Write a haiku about GPUs, then explain what a GPU is in two sentences."   # smoke: coherent text
```
Pass: ctest green; G2; G4.

### P4 -- TpModel on two GPUs: prefill, greedy, sampled, CLI `--tp 2` (~1,100 LOC)

| File | Change | LOC |
|---|---|---|
| `src/model/tp_model.cpp` | `kReal`: mailbox allocation + zeroing on rank 0, device validation (2.9 steps 3-4), recovery path; lift the `kReal` rejection | +260 |
| `src/cli/cli_args.h`, `main.cpp` | enable `--tp-mode real`; `--stats` tp line | +60 |
| `tests/model/tool_tp_soak.cpp` | 10.5 | 440 |
| `tests/model/test_tp_real_vs_emulation.cpp` | 10.1 | 320 |
| `README.md`, `docs/perf.md` | TP section + measured numbers | docs |

Gates (production server on device 1 stopped):
```powershell
.\tests\run_tests.ps1 -TwoGpu                                   # incl. test_tp_real_vs_emulation
Remove-Item env:HIP_VISIBLE_DEVICES
# G6: section 10.4 "G6 [P4]" block (regenerates the emulation reference with this binary)
# G7 (twice; plus 3 identical greedy SHA-256):
build\win-hip\src\cli\r4dx-cli.exe --model D:\models\r4dx\qwen38-27b-v6.r4dx --layout w4a16 --vision off `
  --think off --temperature 0 --max-tokens 256 --max-ctx 2048 --stats --tp 2 `
  --prompt "Write a haiku about GPUs, then explain what a GPU is in two sentences."
# sampled reproducibility: same seed twice -> same text
build\win-hip\src\cli\r4dx-cli.exe <same> --temperature 0.7 --top-k 20 --top-p 0.8 --seed 1
build\win-hip\tests\model\tool_tp_soak.exe --model D:\models\r4dx\qwen38-27b-v6.r4dx --layout w4a16 --minutes 60 --max-ctx 8192 --json build\logs\tp_soak.json   # G8
$env:HIP_VISIBLE_DEVICES='1'; powershell -File tools\tp\tp1_identity.ps1 -Baseline build\baseline -Candidate build\win-hip  # G2
```
Pass: G6 all hashes equal; decode >= 52.2 tok/s in both runs; 3/3 greedy SHA equal; seeded sampled
runs equal; G8; G2. If G7 misses: below 1.40x (50.4 tok/s) stop, record, do not start P5 (risk
R1/R2); between 1.40x and 1.45x record the numbers and the user decides whether P5 starts.

### P5 -- MTP, DFlash2, vision, server under TP (~1,500 LOC)

| File | Change | LOC |
|---|---|---|
| `src/model/mtp_head.{h,cpp}` | 8.1: comm, `logits_rows` sizing, `prime_attn_layer_` (comm null), H6 per-step merge | +140 |
| `src/model/dflash_draft.{h,cpp}` | 8.2 + `LoadHostCodebooks`; H7 | +170 |
| `src/model/model.cpp`, `model.h` | pass comm/vocab to MtpHead and DflashDraftOptions; `VisionSpliceEnabled` and the 8.3 check table; splice copy kind; `R4DX_TP_TESTING` debug hooks (`DflashDebugLastTop16`, MTP equivalent) | +90 |
| `src/model/container.cpp` | vision config parse on rank > 0 (if not done in P2a) | +30 |
| `src/model/tp_model.cpp`, `text_model.h` | `EncodeImages` solo command, `ImageRows::host`; `R4DX_TP_FAULT` parse; lift the MTP/DFlash/vision rejections | +110 |
| `src/cli/main.cpp` | drop staged rejections; `embeds_on_host` from `ImageRows` | +40 |
| `src/server/server_args.h`, `engine.h/.cpp`, `main.cpp` | `--tp*` flags, `EngineOptions::tp`, `unique_ptr<TextModel>`, `ImageRows`, `ms.embeds_on_host`, per-request log gets `tp=2` | +170 |
| `src/server/prefix_state.h` | `needs_reset_` (8.4) | +10 |
| `tools/server/smoke.ps1` | `-Tp`, `-TpMode`, `-TpFault`; with `-Tp 2` remove `HIP_VISIBLE_DEVICES`, pass `--tp 2` to the server AND to every `r4dx-cli` comparison run it makes. `-TpFault`: start the server with `R4DX_TP_FAULT=1:3000:1` (rank 1 stalls 700 ms at its 3000th post-warm-up all-reduce, i.e. inside the first long request), send request A (expect HTTP 500 or 200 -- either is accepted, it is the fault request), then request B = the standard smoke prompt, which must return 200 with text equal to the non-fault reference | +90 |
| `tools/validate_dflash.ps1`, `tools/validate_spec_sampling.ps1` | `-Tp 2`: remove `HIP_VISIBLE_DEVICES` and pass `--tp 2` to every `r4dx-cli` run (ground truth, speculative and control runs alike) | +40 |
| tests: `test_tp_emulation.cpp` (+420: MTP, DFlash, tiny-T), `test_tp_real_vs_emulation.cpp` (+180: MTP, DFlash), `tests/server/test_prefix_state.cpp` (+25) | 10.1 | 625 |

Optional in P5 (behaviour-identical, speed only): the server's plain greedy loop (`engine.cpp:733-759`)
calls `DecodeStep` + host `Sample`; switch it to `DecodeStepGreedy` (same token: device argmax and host
`Argmax` share the lowest-index tie-break) to avoid a 1 MB gather per token under TP. Must pass G2.

Gates:
```powershell
Remove-Item env:HIP_VISIBLE_DEVICES
$cli = "build\win-hip\src\cli\r4dx-cli.exe"; $m = "D:\models\r4dx\qwen38-27b-v6.r4dx"
$d = "D:\models\r4dx\qwen38-27b-dflash2-w4a16-g64.r4dx"
# G9 (twice)
& $cli --model $m --layout w4a16 --vision off --think off --temperature 0 --max-tokens 256 --max-ctx 2048 `
  --stats --tp 2 --dflash $d --dflash-k 7 --prompt "Write a haiku about GPUs, then explain what a GPU is in two sentences."
# G10: each line of tests\model\mtp_prompts.txt, --tp 2 and (HIP_VISIBLE_DEVICES=1) --tp 1, same flags;
#      mean "tok/round avg" of the 4 prompts at TP=2 within +-5% of TP=1
# G11
.\tools\server\smoke.ps1 -Tp 2
.\tools\server\smoke.ps1 -Tp 2 -Model $m -Layers -1 -Dflash $d -ToolRoundTrip
.\tools\server\smoke.ps1 -Tp 2 -Model $m -Layers -1 -Vision -Dflash $d
.\tools\server\smoke.ps1 -Tp 2 -Model $m -Layers -1 -Mtp 3
.\tools\server\smoke.ps1 -Tp 2 -Model $m -Layers -1 -Dflash $d -TpFault       # recovery on the production config
# G12
.\tools\validate_dflash.ps1 -Tp 2 -Layouts w4a16 -AllowBatchedVerifyDivergence
.\tools\validate_spec_sampling.ps1 -Tp 2 -Layouts w4a16 -AllowBatchedVerifyDivergence
.\tests\run_tests.ps1 -TwoGpu
$env:HIP_VISIBLE_DEVICES='1'; .\tests\run_tests.ps1; powershell -File tools\tp\tp1_identity.ps1 -Baseline build\baseline -Candidate build\win-hip
```
Pass: >= 95 tok/s twice; G10; every smoke run 0 FAIL, including `-TpFault`; both validate scripts
exit 0 (a WARN row is accepted only through the scripts' own exact-hash control rule, and every WARN
is listed in the phase commit message); ctest green (incl. the DFlash/MTP cases of
`test_tp_emulation` and `test_tp_real_vs_emulation`); G2. `--mtp 3` tok/s is recorded in
docs/perf.md (projected ~100-110, not gated).

---

## 12. Risks and mitigations

| # | Risk | Mitigation |
|---|---|---|
| R1 | In-engine AR latency exceeds tp_bench's (dirty L2 from GDN state writes, launch placement, inter-card skew at 128 barriers/token) | G5 measures it in the decode pattern with dirty L2 before any model work depends on it; P4 stops at < 1.4x |
| R2 | Device 0 (desktop) is slower or jittery; every AR waits for the slower card | rank 1 = device 0 so the vision encode and rank-0 extras land on the headless card; P2a records rank 1's step on device 0 vs device 1 before any integration work; soak records max exchange wait; if skew dominates, test with the desktop idle (or the display moved off device 0, if the machine allows it) |
| R3 | Per-rank step misses G3 (0.633 x TP=1, ~17.6 ms): halved GEMMs on untuned rows, occupancy-bound GDN kernels (recurrent_update fused-norm = 24 WGs/rank), per-GEMM ~4.5 us intercepts that do not halve | G3 is measured in P2a, before any threading/emulation code exists; P2a tunes every TP2 shape before measuring; if still over, try the non-fused recurrent path with V-splits (more WGs) + separate `gated_rmsnorm`, and bf16 `attn.k/v` at N=512 with a wider WV; record the per-kernel breakdown with `--profile` at TP=1 vs the rank shard run as TP=1 (profiling under TP is disabled) |
| R4 | Lockstep divergence (a data-dependent branch on local data) | SPMD rules L1-L7, H1/H2 fingerprints, result + rng reconciliation every command; worst case a 500 ms timeout, never a hang or silent output |
| R5 | Silent data corruption across PCIe (the `hipMemcpyPeerAsync` class of bug) | only the kernel-driven protocol validated by 12.28M ARs; 10M-AR engine stress (G5); real-vs-emulation byte identity (G6) on every run of the corpus |
| R6 | Windows TDR / desktop freeze from a spinning kernel on device 0 | 500 ms hard timeout (cap 1500) + sticky abort + host abort words; soak checks event 4101 |
| R7 | Implicit device syncs (hipFree, lazy hipMalloc) stalling a lockstepped rank | no allocation or free inside a collective command after warm-up (solo `EncodeImages` excepted, 6.3.7); `GdnControlCache::Freeze`; allocation guard counter must stay 0 in tests and soak; `ImageRows` pageable (no hipHostFree on the facade thread) |
| R8 | Emulation not bit-identical to real because of a device-global resource | `topk_lse` workspace variant; audited: no other `__device__` globals in `src/kernels` or `third_party/libr4d`; G6 catches anything missed |
| R9 | Host-side overhead of rendezvous and thread wake-ups on Windows | spin-then-block waits (20 ms rank, 200 us facade, 50 ms exchange); expected < 0.1 ms/token; `TpCommStats` reports max wait |
| R10 | w4a8/mxfp4 numerics drift more under TP (local-K activation scales) | G4 is on the production layout (w4a16); w4a8/mxfp4 get the looser rel-L2 check in `test_tp_emulation` and an informational KL report |
| R11 | Prefill gains below projection (640 KiB ARs, 58 us x 128 per chunk) | channel 1 `nb` sweep in P3; long-context prefill is attention-bound and gains little by design (1.4) |
| R12 | VRAM: two replicated drafters + replicated embedding + desktop on device 0 | 4.5 shows ~17 GiB/rank at 262k with everything on; the joint embed-residency decision falls back to host gather on both ranks if either is short |
| R13 | Recovery cannot run (a stream stuck past the watchdog) | `kFatal`: every call throws; the server answers errors and must be restarted; logged with rank and Status |
| R14 | Scope creep into graphs / fused AR+norm / drafter sharding | explicit non-goals (1.2); each is a separate measured proposal after P5 |
| R15 | `tools/server/smoke.ps1` compares server output against `r4dx-cli` output | P5 passes `--tp 2` to both sides of every comparison |
| R16 | The server never recovers from a TP error (toggle before `Reset()`; `PrefixState::Invalidate` never forcing `Reset()`) | host-only calls legal in `kNeedsRecovery` (2.4); `needs_reset_` (8.4); G11 `-TpFault` on the production `--dflash` config; `test_tp_emulation` checks the toggle and accessors in `kNeedsRecovery` |
| R17 | Lost wakeup or a data race in the facade/rank hand-off (C++17 has no atomic wait) | publish-under-mutex protocol (2.2); `test_tp_rank_worker` runs 1M hand-offs with spinning disabled; shutdown never writes `cmd_` of a busy worker |
| R18 | First-request skew (lazy module loads, tuning-cache fills, first page faults) trips the 500 ms timeout | warm-up runs every enabled path once under a 1500 ms timeout, then `Reset` + `Freeze` (2.9 step 9) |
| R19 | A two-GPU test lands on the desktop GPU or on device 1 next to the production server | `R4DX_TP2GPU` opt-in + `ctest --test-dir` without the preset + server/VRAM pre-flight (10.1, 9.2) |
| R20 | TP rows change TP=1 kernels (the drafter's `17408 x 5120` gate/up) | separate TP table consulted only on TP threads (2.7); G2 rows 2 and 5 cover the drafter |

---

## Appendix A -- file map

New: `src/core/include/r4dx/core/tp_comm.hpp`, `tp_host_exchange.hpp`, `tp_alloc_guard.hpp`;
`src/kernels/include/r4dx/kernels/tp_kernels.h`, `src/kernels/src/r4dx_tp_kernels.hip`,
`src/kernels/check_tp_isa.cmake`; `src/model/tp/tp_shard.{h,cpp}`, `tp_vocab.h`, `tp_rank_worker.h`,
`tp_group.{h,cpp}`, `tp_comm_noop.{h,cpp}`, `tp_comm_emulated.cpp`, `tp_comm_host_mailbox.cpp`;
`src/model/model_types.h`, `text_model.h`, `local_text_model.{h,cpp}`, `tp_model.{h,cpp}`,
`gemm_tuning_table_tp2.inc` (generated); tests and tools listed in 10.1;
`tools/tp/tp1_identity.ps1`.

Modified: `src/model/model_config.h`, `container.{h,cpp}`, `model.{h,cpp}`, `gdn_layer.{h,cpp}`,
`mlp.{h,cpp}`, `gdn_state.h`, `linear.{h,cpp}`, `mtp_head.{h,cpp}`, `dflash_draft.{h,cpp}`,
`attn_config.h`, `attention/include/r4dx/model/attention/{types.hpp,attention_layer.hpp}`,
`src/model/CMakeLists.txt`; `src/core/include/r4dx/core/device_buffer.hpp` (guard hook);
`src/kernels/src/r4dx_kernels.hip`, `kernels.h`, `CMakeLists.txt`; `src/vision/image_prompt.h`;
`src/cli/{cli_args.h,main.cpp}`; `src/server/{server_args.h,engine.h,engine.cpp,main.cpp,
prefix_state.h}`; `tests/model/{teacher_forced.h,tool_teacher_forced_logprobs.cpp,CMakeLists.txt}`,
`tests/server/test_prefix_state.cpp`, `tests/kernels/CMakeLists.txt`, `tests/core/CMakeLists.txt`,
`tests/run_tests.ps1`; `tools/profile/tune_gemm.py` (`SHAPES` only); `tools/server/smoke.ps1`,
`tools/validate_dflash.ps1`, `tools/validate_spec_sampling.ps1`; `README.md`, `docs/perf.md`.
**Not** modified: `src/model/gemm_tuning_table.inc`.

CMake targets: `r4dx_tp_shard` (CPU), `r4dx_tp` (links `r4dx_core`, `r4dx_kernels`),
`r4dx_model` links `r4dx_tp`; `r4dx_kernels` gains the second hipcc object and the ISA check.

Estimated total: **~12,000 LOC** including tests (P1 2,400; P2a 1,650; P3 2,650; P2b 2,700;
P4 1,100; P5 1,500). The revision added ~1,700 LOC over the first draft, mostly tests (rank
worker, DFlash/MTP exactness, fault injection) and the recovery/warm-up hardening.

---

## Appendix B -- Rejected review notes

Each entry: what a reviewer asked for, why this design does not do it (or does something else),
and the evidence.

- **B1. "G3 measures the step with no kernel in the all-reduce slot, but the projection adds
  tp_bench's vs-stand-in L, so the projection is short by 128 x the stand-in cost."** Rejected in
  its premise. The 7.47 us (10 KiB) and 13.33 us (80 KiB) numbers this design uses **are**
  tp_bench's `L_vs_no_ar_kernel` values, not vs-stand-in: the sweep's own output
  (`C:\Users\pay20\dev\r4dx\tools\tp_bench\build\runs\sweep.json`, untracked build output in the
  main worktree, 2026-09-24) has in `summary.decision` `best_L_vs_no_ar_kernel_us_10KiB = 7.470` and `_80KiB = 13.333`, while
  `best_L_us_10KiB` (vs stand-in) is 8.263 and `_80KiB` 11.939; `tools/tp_bench/README.md:23` labels
  them "(vs no AR kernel)". `NoopComm` with no kernel is exactly tp_bench's condition (c), so G3 +
  128 x 7.47 us is the consistent projection, as the `baseline_note` in `ar_bench.hip:3340-3343`
  prescribes. Accepted from the same finding: G3 is now a ratio to the same tool's TP=1
  measurement (not an absolute ms derived from CLI numbers), and G5 / P3 / `tool_tp_ar_latency`
  now say explicitly which L is gated (the first draft's P3 text wrongly said "vs stand-in").
- **B2. "Gate: `--tp 2 --dflash` greedy stdout SHA == `--tp 2` plain greedy SHA."** Not adopted as a
  hard gate, because the equality does not hold at TP=1 either: batched-verify reduction order
  legitimately changes greedy text on some prompts (`docs/dflash2.md` 814-835: w4a16 "medium"
  mismatches, reproduced with the identical SHA by an `--mtp 7` control that involves no DFlash
  code). Instead G12 runs the project's own losslessness gates, `tools/validate_dflash.ps1` and
  `tools/validate_spec_sampling.ps1`, with `-Tp 2`: plain vs speculative equality, and a mismatch
  is accepted only when a control reproduces the exact hash. `test_tp_emulation` adds exact checks
  of the TP-specific code: the H7 merge against a gathered full-row top-16, the H6 merge,
  per-round bookkeeping, and emulation-vs-real byte identity of DFlash rounds.
- **B3. "Run P3 in parallel with P2a."** Adopted as an **ordering** (P1 -> P2a -> P3 -> P2b), not as
  concurrent work: concurrency would need a second worktree/branch and a merge, and the goal --
  both economic gates known before P2b -- is met by the order alone.
- **B4. "Give `AttentionLayer::Forward` a `kv_only` flag so `PrimeKv` skips `o_proj` (also at
  TP=1)."** Not adopted for v1: it changes TP=1's launch sequence, which G2 would have to re-prove.
  `PrimeKv` instead uses a second `AttentionLayer` built with a null comm (6.2), which removes the
  wasted all-reduce under TP and leaves TP=1 untouched. Skipping the discarded `o_proj` GEMM at
  TP=1 too is a separate speed proposal.
- **B5. "Or: any call in `kNeedsRecovery` runs Recovery + Reset implicitly."** Not adopted. An
  implicit reset would hide that the sequence state was dropped from a caller that believes it is
  extending a prefix. The explicit rule plus the two fixes in 2.4 / 8.4 make the server recover
  without that ambiguity.
- **B6. "Re-measure TP=1 DFlash baselines after adding the TP tuning rows, or land the sweep on
  main first."** Superseded: the rows now live in a TP-only table (2.7), so TP=1 bytes and timing do
  not change and the production G9/G10 references stay valid.

### Implementation notes

Where the code refined or corrected this design while implementing it. Each note names the section
it refines.

**P1**

- **N1 (2.2 step 6).** `ProgressWatchdog` compares each rank's heartbeat with its value at the
  previous wake-up, not the maximum over ranks: the maximum can stand still while a lagging rank
  catches up, and that is progress. It fires on exactly the stated rule: no rank's heartbeat has
  moved for 60 s. The step-5 facade wait is `WaitAllIdle(workers, group, timing, watchdog, wake,
  now)` in `tp_rank_worker.h`. It waits for `Idle()` on every listed worker (the facade is the only
  poster, so that equals `Done() >= seq`). It returns `kStalled` instead of throwing, so `TpModel`
  keeps ownership of `kFatal`, the log line and `TpTimeoutError`. `now` injects the test's fake
  clock.
- **N2 (2.2 step 1).** `RankWorker::Post` to a busy worker throws `std::logic_error` instead of
  asserting (the `assert` would be compiled out in the Release build). If `thread_init` throws (a
  failed `hipSetDevice`), the worker stays up. Every command then completes without running, and
  its `TakeError()` returns that exception.
- **N3 (5.2 `RuleFor`).** Replicated names are listed explicitly instead of matched by the
  `*layernorm` / `*norm*` wildcards, so a future tensor whose name merely contains "norm" still
  throws until someone classifies it. `dflash.*` replicates (4.2's DFlash2 row; 5.2's list omits
  it). `mtp.draft_head.*` is the two exact names `mtp.draft_head.lm_head` and
  `mtp.draft_head.vocab_ids`; `vision.*` (rank 0 only) and `dflash.*` stay whole-prefix rules, as
  4.2 defines them. `RuleFor` also rejects: a layer index >= `num_hidden_layers`; a GDN tensor on a
  full-attention layer, or the reverse; `mtp.gdn.*`; and, on every path including the two prefix
  rules, any name that still carries its `.{bf16|w4a16|w4a8|mxfp4}.{part}` suffix.
- **N4 (4.3, 5.1 step 4).** `PlanRows`/`PlanCols` merge adjacent byte runs, so "one contiguous
  range" means `runs.size() == 1`. `test_tp_shard` confirms 5.1's list on the real shapes: every
  part of qg, z, k/v and lm_head except mxfp4 ws; a/b, A_log, dt_bias and the descales; and mxfp4
  ws K-slices. mxfp4 wq column ranges need % 32 (4.3's rule, from the mxfp4 group), although wq
  packing alone needs only % 16. Both planners also validate the full shape (N % 16; K % 64,
  % group or % 32), so a malformed shape throws instead of truncating.
- **N5 (10.2).** "K = 1088 (= 17 x 64) at g=64" is the per-rank K, so the full K is 2176. A full
  K of 1088 would give 544 = 8.5 x 64 per rank, which no w4 layout can split. At full K 2176 the
  test checks bf16, w4a16 g64 and mxfp4 byte for byte, and checks that the four group-128 layouts
  (w4a16 g128 and w4a8, each RTN and search) are refused. The end-to-end slices use a small config
  with the real structure: hidden 512, 4 q / 2 kv heads of 256, GDN 2 k / 8 v heads of 128,
  intermediate 1024, vocab 1024. Plan-only checks cover the real 27B shapes: size, legality and
  contiguity for every linear x layout x rank.
- **N6 (10.1 `test_tp_vocab_merge`, 7.4).** "lse within 1e-6" is replaced by an ABSOLUTE bound of
  2 fp32 ulp of the exact double-precision full-row logsumexp, plus 1e-7. A flat 1e-6 would fail
  an exact merge on rounding alone (one ulp is 1-2e-6 at |lse| ~ 10-20). A bound relative to |lse|
  is wrong too: `SampleFromSummary` reads lse only through `S_full = exp(lse - vals[0] * inv_t)`,
  so an absolute lse error is the relative `S_full` error that `kRowSummaryLseRelTol = 1e-3`
  bounds. At the tested T = 0.005, |lse| reaches 2000-2700, and 1e-6 * |lse| would accept 2-3x the
  whole band. An exact merge carries two fp32 roundings (the dominant shard's lse and the merged
  value) and lands within ~1 ulp; 2 ulp is <= 4.9e-4 on these rows, and the test refuses a row
  where 2 ulp would reach the band. The same accounting corrects 7.4's "+1e-7". The 100000-u sweep
  runs at V = 4096. At V = 248320 it is a 400-u spot check, because `SampleCanonical` costs
  milliseconds per call there.
- **N7 (6.6).** A size mismatch between ranks aborts the group and throws `TpDivergenceError`.
  `Barrier` is a 0-byte `AllGather`, so both share one generation counter. After the 50 ms spin,
  the wait loop calls `std::this_thread::yield()`, not `Sleep(0)`, which would pull `<windows.h>`
  into a src/core header. `tp_comm.hpp` includes `<hip/hip_runtime_api.h>` for `hipStream_t`, so
  `test_tp_host_exchange` links `r4dx_core` for the include path. It makes no HIP call and imports
  no HIP DLL.

**P1 review**

- **N8 (2.2 step 6, 2.9).** `ProgressWatchdog` compares elapsed time in whole milliseconds. The
  plain `now - last_move_ >= stall_limit` converted the limit to the clock's nanoseconds and
  overflowed for any limit past ~292 years, so `milliseconds::max()` fired at the first wake-up.
  `ProgressWatchdog::kNoStallLimit` (= `milliseconds::max()`) now never fires. A command that moves
  no heartbeat -- 2.9's `LoadEmbedTokensHost` and `Model::Load`, the teardown closure -- must be
  waited on with `kNoStallLimit` or a limit sized for it: the 60 s default would kill any load
  that takes longer (TP=1 bf16 loads take 58-64 s today).
- **N9 (6.6).** `HostExchange::AllGather` checks the abort flag on entry, before `++gen` and the
  slot write. A call on an aborted group never reaches the wait, so it must not rewrite
  `slot[g & 1]` either: a slower peer may still be copying call g's slot, and the parity argument
  assumes the writer passed call g + 1's wait. `Abort` sets the flag inside the critical section
  that tests it, so the first reason really is the one kept.
- **N10 (3.2, 5.1 step 7).** `ModelConfig::Shard` applies its TP-only checks (no tied embeddings,
  row-parallel K % 512, vocab % 16) at world 1 too, so the TP=1 load path must not call it:
  `config_ = global_config_` there. 3.2's "at world 1 the two are equal" now says so.
- **N11 (10.2).** `TestRealPlans` also checks, at the real 27B shapes, that every plan's runs are
  ascending, disjoint and inside the full part, and pins exact offsets from 4.3's formulas for
  lm_head bf16 and mxfp4 ws, mlp.down w4 wq and w4a16 g64 wsz, and mlp.gate_up w4a8 ws and mxfp4
  wq, all on rank 1. The small config keeps every offset under 2 MB, so only these checks would
  catch a narrowed intermediate.

**P2a**

- **N12 (3.3, 5.1 step 7).** `Container::Load(path, ContainerLoadOptions)` at `tp_world == 1`
  runs the pre-TP loader body. The only additions there are the options unpack and
  `global_config_ = config_`. It refuses the TP-only options `shared_embed_host`,
  `embed_device_resident_decided >= 0` and `parse_vision_config` at world 1 with
  `std::invalid_argument` rather than ignoring them. The shard path is a separate private
  `Container::LoadShard` that walks the same tensor set, so the TP=1 path does no rule lookup and no
  staging. `load_vision` on a rank other than 0 is refused (4.2: rank-0-only).
- **N13 (4.3, 5.1 step 4).** Layout-less row-split tensors (`gdn.in_proj_a/b`, `conv1d_weight`,
  `A_log`, `dt_bias`, the descales) use one planner path: `Part::kElem` with
  `row_bytes = span / (rule rows)`. That covers 2-D bf16, conv1d and fp32 vectors alike. Before
  slicing, every quantized or bf16 part's span is checked against the byte size its global `[N, K]`
  implies, so a malformed tensor throws instead of being sliced with the wrong strides. The loader
  gathers into one reusable staging vector rather than calling `tp::Gather`, which allocates on
  every call. For the embedding device mirror, `embed_device_resident_decided >= 0` overrides both
  `embed_device_resident` and the free-VRAM heuristic, because the joint decision (2.9 step 5)
  already includes the flag.
- **N14 (2.9 step 1, 3.3).** The staged rejections are also enforced in `Model::Load` when
  `tp.world > 1`. `mtp_draft_k > 0`, a non-empty `dflash_container` and `vision == kOn` throw
  `TpUnsupportedError`; `kAuto` loads text-only and logs a line. This stops a bare rank `Model`,
  which is what `tool_tp_step_bench` drives before `TpModel` exists, from reaching a path whose TP
  hooks are not in yet. `Model::Load` also validates `TpRankOptions`: the TP-only fields must keep
  their defaults at world 1, and `comm->World()/Rank()` must match `world/rank`.
  `DflashHostCodebooks` is only forward-declared in `model.h`, and `tp.dflash_codebooks` stays null
  until P5.
- **N15 (P2a "sampled/speculative throw").** These entry points throw `TpUnsupportedError` on a TP
  rank:
  - `DecodeStepSampled`, only when it would actually sample (`temperature > 0`); the greedy
    routing is the plain path and works;
  - `VerifyWindow`, `CommitVerifiedWindow`, `ReadVerifyLogitsRow`;
  - `DecodeStepMtp*` and `DecodeStepDflash*`, through their shared `*Impl`;
  - `DecodeStepProfiled` and `PrefillProfiled`.

  `RunChunk` additionally refuses a summary request under TP with a `std::logic_error`, which no
  public path can reach.
- **N16 (6.2 H1).** The fingerprint's `kind` word is `"RUNC" << 32` OR'd with the call's mode
  bits: 1 = prefill path, 2 = logits wanted, 4 = greedy. Each mode issues a different collective
  sequence after the layers, so a mode disagreement now fails at H1 rather than at a mismatched
  all-gather.
- **N17 (4.4, 7.2, 7.5).** `argmax_pair_dev_` is a `DeviceBuffer<int32_t>` of `2 x draft_window_`
  elements. Row t holds `{int32 local idx, float bits}`, byte-compatible with `tp::ArgmaxPair`
  (static_assert). `GatherVocabRow` copies the shard into its own host vector first, so the
  all-gather's source never aliases its destination.
- **N18 (6.4).** `NoopComm::AllReduceSumBf16` checks its arguments the way the real transport will
  (non-null, `0 < 2n <= 655360`, `2n % 16 == 0`). It also counts per-channel `Stats()` and
  `CallCounts()`, so `tool_tp_step_bench` can report how many all-reduce sites a step hit (8 per
  token on a 4-layer model, 128 on 64 layers). `Abort` only records the abort. `SelfTest` and
  `SetAllReduceTimeoutMs` are no-ops.
- **N19 (2.7, 10.1).** `test_pick_tuning` now also runs with `SetTp2TuningForThisThread(true)` and
  checks every row of `gemm_tuning_table_tp2.inc`, plus the main table behind it, for launchability
  at the build's groups. This addition is not in 10.1's list. It matters because the TP table has
  `SK=16` rows at `K=5120`, which are legal only at group 64: the hazard that test exists for.
- **N20 (P2a tuning sweep).** Both sweeps ran on device 1 with the group-64 pyd, exactly as 11
  gives them, and wrote 42 w4a16 rows and 7 bf16 rows. The generated file's header line names only
  `w4a16=64`, because `--append` does not rewrite the header. Measured M=1 times in us (for R3):

  | Row | M=1 time (us) |
  |---|---|
  | `tp2.gdn.in_proj_qkv` | 27.0 |
  | `tp2.gdn.in_proj_z` | 18.6 |
  | `tp2.out_proj` | 18.0 |
  | `tp2.mlp.gate_up` | 76.3 |
  | `tp2.mlp.down` | 40.8 |
  | `tp2.lm_head` | 567.8 (7.1 estimated ~550) |
  | `tp2.attn.kv` (bf16) | 13.1 |
- **N21 (10.1 `tool_tp_step_bench`).**
  - It runs one untimed warm-up pass (`--warmup 16` decode steps) before the `--repeats` timed
    passes. The first pass would otherwise absorb lazy module loads and `PickTuning` cache fills.
  - Pre-flight defaults: `--need-gib` is 17 for `--tp1` and 11 for one rank.
  - `--layers N` sets `layer_limit`, for smoke runs on the 4-layer containers; G3 omits it.
  - The JSON has `rank = -1` for `--tp1`, and adds `mode`, `hip_device`, `model`, `layout`,
    `max_ctx`, `layers`, `tokens`, `repeats` and `prompt_tokens` to 11's fields.
- **N22 (10.3 `tp1_identity.ps1`).** The script captures each binary's stdout byte for byte
  through `System.Diagnostics.Process`, because PowerShell 5.1's `>` re-encodes native output. It
  also:
  - finds binaries in a flat directory (`build\baseline`) or a build tree (`build\win-hip`);
  - takes `-Rows` to select rows;
  - passes `--quiet` to the teacher-forced tool;
  - writes every artifact and a `summary.txt` under `build\logs\tp1_identity`;
  - throws on any DIFF or failed run.
- **N23 (10.1 `test_tp_loader`).**
  - Both ranks of a layout are loaded side by side (peak ~7 GiB).
  - Every buffer is compared with `Gather(Plan(...))` of the file.
  - Every rank shape is checked against 4.2's literal numbers, not re-derived from the rules.
  - The bf16 reassembly maps every global row, or every row's column range, to its (rank, local
    offset) by index arithmetic alone and compares it in place, without building the full tensor.
    It covers all 47 sharded tensors of the 4-layer container.
  - The embedding mirror is checked with decision 1 on the w4a16 pass and 0 on the others.
  - The load-time refusals of N12 are checked.
  - First run: 1675/1675 checks.

**P2a review**

- **N24 (2.7).** `Model::Load` calls `SetTp2TuningForThisThread(tp.world > 1)` on **every** load,
  not only `true` on a rank load, and a guard clears the flag again when the load throws. The flag
  therefore follows the thread's latest successful load: a TP=1 `Model` loaded on a thread that
  loaded a rank earlier (a NoopComm rank, then a TP=1 reference, in one tool or test) resolves from
  the main table alone, so its DFlash drafter's `(w4a16, 17408, 5120)` stays on `FallbackTuning`.
  An RAII guard around every forward entry point was the alternative. It was not taken because it
  needs one guard per entry point (`RunChunk`, `VerifyWindow`, the MTP, DFlash, `PrimeKv` and
  profiled paths), and a missed one silently mixes tables. What remains: a rank `Model` used after
  a later TP=1 load on the same thread runs on main-table tunings. That is slower but produces the
  same bytes a rank would with any legal tuning. `TpModel` loads each rank on its own thread, so
  this does not arise there. The same review also extends N14's world-1 refusal to
  `tp.vision_weights_on_this_rank = false`.
- **N25 (P2a tuning sweep, 2.7).** `tune_gemm.py` enforces the table split. An empty `--shapes`
  selects every shape except the `tp2.*` ones, which run only when named. A run refuses to write
  `tp2.*` rows to any `--out` other than `gemm_tuning_table_tp2.inc`, and other rows to that file.
  Before this, the documented full regeneration (`--out src\model\gemm_tuning_table.inc`, or
  `--layouts w4a16 --replace` after a group change) swept the TP shapes too. That put
  `(w4a16, 17408, 5120)` into the main table (TP=1 DFlash bytes change), or it aborted the
  `--replace` after the whole sweep.
- **N26 (10.3).** `tp1_identity.ps1`:
  - On every CLI row it also compares the timing-free stderr lines `[stats] mtp:`,
    `[stats] dflash:` and `[stats] sampled:` (rounds, drafted, accepted, fallback rows) byte for
    byte. Greedy verify and seeded sample-and-match emit tokens that do not depend on the drafts, so
    text and ids alone cannot see a change in the TP=1 MTP-head or drafter bytes, which is the
    change the separate TP table exists to prevent.
  - It adds row 8, row 4 + `--mtp 3` (MTP sampled rounds), and row 9, a `--chat` session of two
    user turns fed through stdin, greedy (the second turn prefills a suffix on top of the first
    turn's state). A server row is not added. It needs an HTTP driver, and the server's
    Reset/suffix path is the CLI's `Model` calls.
  - A SKIPped row (row 7 without the golden image) makes the run `G2 INCOMPLETE` and exits
    non-zero unless `-AllowSkip` is given.
  - The tp2 worktree has no gitignored `golden_out`. G2 therefore passes
    `-Image C:\Users\pay20\dev\r4dx\tools\reference\golden_out\vision_test_image.png`, which only
    reads it.
- **N27 (10.1 `tool_tp_step_bench`).** The tool prints and writes `embed_device_resident` to the
  JSON. Each process takes the embedding-mirror decision from its own free VRAM, so G3 compares
  like with like only when the TP=1 and rank JSONs agree, and the P2a G3 run checks that they do.
- **N28 (2.7, P2a tuning sweep).** `gemm_tuning_table_tp2.inc` is tuned for w4a16 group 64 only.
  At group 128, `BestRow` skips the `SK=16` `tp2.mlp.gate_up` rows for M=1..32 and rounds up to the
  M=64 row, a prefill-shaped config, for decode. A group-128 TP build must re-sweep the TP table at
  its group before any TP measurement means anything.
- **N29 (P2a gates, measured 2026-09-24 after the review fixes, device 1, production server
  stopped).**
  - `run_tests.ps1`: 70/71 pass (`test_tp_loader` and `test_pick_tuning` included). The one
    failure is `test_mtp` `CheckSampledRoundsMatchPlain [w4a16]` (0/18 identical sampled
    trajectories). It is a known failure that predates P2a; the suspected cause is `main`'s
    `32093f2` GEMM re-sweep.
  - G2: `tp1_identity.ps1` rows 1-9 all EQUAL, with row 7 on the golden image through `-Image`
    (N26). Row 9's second turn prefilled a 24-token suffix at position 37 without a re-render
    reset.
  - G3 (`tool_tp_step_bench`, median of 3 x 128 tokens; every run mirrored `embed_tokens` on the
    device):

    | Run | Median (ms/token) | Ratio to TP=1 |
    |---|---|---|
    | TP=1 | 27.644 | |
    | rank 0 | 15.115 | 0.5468 |
    | rank 1 | 15.159 | 0.5484 |

    Both ranks are at or below 0.633, so G3 passes. Rank 1 is 1.5-1.6 ms under 1.4's
    16.7-17.6 ms model.
  - Recorded, not gated (R2): rank 1 on device 0 (the desktop card, pci bus 3) measured
    15.243 ms/token. That is +0.084 ms (+0.55%) over device 1.

**P3**

- **N30 (6.3.1 `R4dxTpArArgs`, 6.3.5 `CheckHealthy`).** `R4dxTpArArgs` has two more fields,
  `int32_t channel` and a padding `reserved`. On a timeout or protocol violation the kernel
  records the channel in `R4dxTpStatus::word`. tp_bench's FLAG variant stored `0xffffffff` there,
  because `word` was LL-only. Without the field, the "at channel <c>" of 6.3.5's message cannot be
  derived. One `Status` per rank is shared by both channels, which keeps the sticky abort
  rank-wide, and both channels' seq counters start at the same base, so a seq value does not
  identify its channel. The `Status` layout is unchanged.
  - The launcher turns `R4dxTpArArgs` into tp_bench's typed-pointer `ArArgs` and passes that by
    value. The backend promotes pointers it loads from the kernel-argument segment to the global
    address space. Casting `int64` to a pointer inside the kernel made every access a generic
    `flat_*` instruction instead of `global_*` (checked in the ISA).
- **N31 (6.3.5 `HostU32Load/Store`).** The two helpers are `inline` in `tp_group.h`, not in
  `tp_comm_host_mailbox.cpp`. That way `TpGroup` and the CPU-peer test use the same helpers as the
  endpoint.
- **N32 (6.3.3, 2.5 step 4, 2.9 step 7): the mailbox reset writes the session base into every
  flag line.** A zeroed flag is "behind" `s = base + 1` under the wrap-safe compare only while
  `base < 2^31`.
  - The wrap case of 10.1 (base `0xFFFFFF00`) failed on its first run. A zeroed flag read as 255
    ahead of `s`, the ">= 2 ahead" branch fired, and that is a protocol violation (the CPU peer
    reported it first).
  - In production the same thing would hit the `SelfTest` of any recovery once the seq counters
    passed 2^31. At 6.3.3's rate that is about 3.5 days of continuous decode, and the result
    would be `kFatal`.
  - The fix: `TpGroup::ResetMailbox(base)` memsets the region, writes `base` into every flag line
    of both channels and both ranks, then issues a `seq_cst` fence. An unwritten flag then reads
    exactly one behind: "not yet". It replaces "zero the whole region" in `AllocateMailbox` and in
    recovery step 4.
  - Stale flags from an older session still compare behind, because `new_base` stays above every
    value used.
- **N33 (2.5 step 1 `SyncWithWatchdog`): it polls a marker event, never `hipStreamQuery`.**
  - Measured on device 1: while `hipStreamQuery` is being polled (a tight `yield` loop, or 1 ms
    sleeps), a stream whose last command is an event record never reports idle. That holds for
    timing and `DisableTiming` events alike, and was still true 3 s after its kernel finished. The
    trailing event's own `hipEventQuery` also stays `NotReady` meanwhile.
  - Recording a marker and polling `hipEventQuery`, as tp_bench's `wait_event_wd` does, completes
    when the kernel ends: 101.6 ms for a 100 ms all-reduce timeout.
  - The wait yields for the first 200,000 polls, then sleeps 1 ms at a time (tp_bench's wait). An
    earlier `sleep_for(200 us)` after 20,000 polls overslept to the Windows timer tick, and that
    made `CheckWallClockRate` measure the device clock 47% slow.
  - Rule for P2b/P4 code: wait on events, never by polling `hipStreamQuery`.
- **N34 (6.3.1 file list, 6.3.5 ISA check, 10.1).**
  - tp_bench's gen, verify, filler, fill-hash and stand-in kernels live in `r4dx_tp_kernels.hip`
    as `r4dx_tp_test_*` entry points. Host twins of the data pattern are in `tp_kernels.h`. The
    tests and tools are clang-cl `.cpp` files, so they need no third hipcc object.
  - The verify kernel accumulates per thread and issues one atomic pair per thread. tp_bench's
    version issued one pair per mismatching element, which took more than 30 s when a whole batch
    was wrong.
  - `tests/kernels/tp_ar_harness.h` is the shared test harness: `RankPool` on `tp::RankWorker`,
    `RunDriver` (tp_bench's `run_plan`/`ar_rank` on a `TpEndpoint`, with the three interleaved
    decode conditions) and `Decode` (tp_bench's `decode_stats`).
  - `check_tp_isa.cmake` requires the following of `r4dx_tp_ar_flag_bf16_kernel` (as revised
    by N43):
    - `.amdhsa_workgroup_processor_mode 1`, and no `-mcumode` in the flags;
    - the flag store, found structurally: the first global/flat/buffer store or atomic after the
      post-push barrier (the first `s_barrier_wait` after the first 16-B push store). It must be a
      32-bit `SCOPE_SYS` store, immediately preceded by `global_wb scope:SCOPE_SYS`, with no store,
      label or branch in between;
    - between the flag store and the next barrier, a `SCOPE_SYS` 32-bit poll load and a standalone
      `global_inv scope:SCOPE_SYS` (the acquire fence after the relaxed spin).
  - The check rejected a hand-edited ISA without the writeback, and one in CU mode. N43 lists the
    revised check's negative cases. The `-S` command adds `-Wno-unused-command-line-argument`,
    because hipcc's link-only flags are unused under `-S`.
- **N35 (build).** `tp_group.cpp`, `tp_comm_host_mailbox.cpp` and the four new test and tool
  sources compile with `/EHc-`. The `r4dx_tp_*` entry points are `extern "C"` and throw on bad
  arguments and HIP errors, and a `/EHsc` caller would terminate instead of unwinding
  (`tests/kernels/CMakeLists.txt` explains why).
- **N36 (6.1, 2.5, 2.9 step 7: the `TpGroup` API).**
  - `TpGroup::Create(mode, world, Geometry{nb_small, nb_large}, ar_timeout_ms, seq_base = 0x1000)`.
    `kEmulate` throws `TpUnsupportedError` until P2b.
  - `AllocateMailbox()` runs on rank 0's thread.
  - `CreateEndpoint(rank, heartbeat)` runs on each rank's thread. It returns a `TpEndpoint`: the
    `TpComm` plus the per-rank recovery pieces (`SyncStreams`, `SeqAdvance`, `Rebase`,
    `ReadStatus`, `ArmFaultInjection`, `Device`, `WallClockKhz`, `SeqBase`).
  - `Recover(RankRunner{run_all, run_one})` runs 2.5 steps 1-7 through the caller's
    `RunAll`/`RunOne`, with `new_base = old_base + max over ranks of (seq - old_base) + 64`.
  - Step 1 syncs only the streams the endpoint knows: its own, and every stream an all-reduce was
    enqueued on, at most 8 distinct ones. An all-reduce on a ninth throws `std::logic_error`
    before enqueuing; nothing is evicted (N43). Those streams must outlive the endpoint's use of
    them, which a `Model`'s `stream_` does. The caller syncs any other stream first, such as
    rank 0's vision stream.
  - `CheckWallClockRate()` (2.9 step 4) is a free function on the current device.
  - `TpOptions::fault_*` maps to `ArmFaultInjection(at, kind)`, which counts from the arming call.
- **N37 (6.3.4 chunking).** Each call launches `nb_eff = ceil(n16 / w)` blocks, with
  `w = ceil(n16 / nb)` (tp_bench's `make_geom`). That equals `nb` for every engine size
  (`n16 >= 640`), and it still depends on the byte count only (L5).
- **N38 (6.3.9 `SelfTest`).** It first syncs the endpoint's streams, because the seq counters are
  stream-ordered: a self-test all-reduce on the endpoint's own stream must not overlap one still
  queued on the model's stream. Its all-reduces count in `Stats()` and `CallCounts()` like any
  other, and both ranks count them identically. A mismatch aborts the group (`kAbortHost`) before
  it throws `TpError`, so the peer fails fast instead of timing out.
- **N39 (10.1 `test_tp_allreduce_cpu_peer`).**
  - The pattern call index is `call % 13`, so the CPU peer keeps every rank's rows precomputed.
    100,000 all-reduces then take 2.5 s instead of about 45 s of CPU hashing. 13 is odd, so data
    stale by 2 calls, the only staleness the slot protocol could produce, still differs from the
    expected row.
  - The 100,000 all-reduces cycle through the four sizes, 25,000 each.
  - The wrap case is 1,200 all-reduces over both channels, from base `0xFFFFFF00`.
  - The bit-exact cases run at the production 500 ms timeout (N43). A CPU peer descheduled for
    longer trips the GPU's timeout, and that is a test failure, not a reason for a longer spin.
  - The timeout case times the kernel with hipEvents and also covers fault injection (kind 0).
    The skipped call is timed on the host, launch to completion: hipEvent timestamps around a
    microsecond-long kernel came out slightly out of order.
  - A fourth case, not in 10.1, checks `r4dx_tp_add_bf16`, EmulatedComm's add for P2b: eight
    640 KiB adds of rank 0's pattern and rank 1's, bit-exact.
- **N40 (10.1 `test_tp_allreduce_2gpu`).**
  - The mixed sizes cycle over `{10240, 81920, 174080, 184320, 655360, 20480, 40960, 122880}`:
    six on channel 0, two on channel 1. There are 50 all-reduces per verified batch.
  - A filler that reads 2 MiB and dirties 1 MiB runs before every all-reduce. tp_bench's 60 MiB /
    4 MiB would make the 1M all-reduces take about 25 minutes instead of 25 s.
  - Abort case: rank 1 waits 50 ms after rank 0 has enqueued, so rank 0's kernel is spinning,
    then calls `Abort()`. The exit delay is measured from the host `steady_clock` at the `Abort`
    to the completion of rank 0's marker event.
- **N41 (10.1 tools).**
  - `tool_tp_ar_stress` options:
    - `--pattern isolated|decode`, `--bytes` (default 10240), `--mixed`;
    - `--nb` / `--nb-large`, `--timeout-ms`, `--filler-mb` / `--dirty-mb`;
    - `--devices` (9.2 auto), `--need-gib` (1.5 for decode, 0.5 for isolated), `--json`.

    It runs `SelfTest` first and uses tp_bench's exit codes.
  - `tool_tp_ar_latency` options:
    - `--tokens` (default 50), `--sizes` (default `10240,81920,655360`);
    - `--nb` (default 4), `--nb-large-list` (default `4,8,16`);
    - `--filler-mb`, `--dirty-mb`, `--timeout-ms`, `--devices`, `--need-gib`, `--json`.
  - The latency tool creates one `TpGroup` per `nb_large` and measures the channel-0 sizes once.
    It prints the G5 limit next to the 10 KiB and 80 KiB rows, and names the best 640 KiB `nb`
    under 11's 10% rule. It exits 0 when every run verified: the thresholds are printed, not
    gated.
- **N42 (P3 smoke, measured 2026-09-24 on both GPUs, production server stopped; not the G5
  gate).**
  - `test_tp_allreduce_cpu_peer` (device 1): passes in 3.0 s.
    - 100,000 all-reduces, bit-exact.
    - The wrap case, bit-exact.
    - Timeout: 100.04 ms on the device for a 100 ms timeout. The `ABORT` word read
      `kAbortTimeout`, and the `Status` named channel 0, block 3, `flag-wait`.
    - The next call was skipped: 4 blocks, 0.06 ms, buffer untouched.
    - The add case, bit-exact.
  - `run_tests.ps1` (default, `-LE tp2gpu`, device 1): 71 of 72 pass, including the CPU-peer
    test. The one failure is the known `test_mtp CheckSampledRoundsMatchPlain [w4a16]` from N29.
  - `tool_tp_ar_stress`: 100,000 isolated 10 KiB all-reduces at 171k per second, and 100,096
    decode-pattern all-reduces at 8.3k per second. All verified.
  - `run_tests.ps1 -TwoGpu`: `test_tp_allreduce_2gpu` passes in 26 s.
    - 1,000,000 mixed all-reduces verified in 25.3 s.
    - After `Abort()`, rank 0's kernel exited 0.068 ms later.
    - `Recover()` moved the base from `0x00001000` to `0x000b8221`.
    - 10,000 clean all-reduces followed.
  - `tool_tp_ar_latency --tokens 10`, a smoke run, not G5. `L_vs_no_ar_kernel` for each case:

    | Size | nb | `L_vs_no_ar_kernel` (us) |
    |---|---|---|
    | 10 KiB | 4 | 5.0 ± 0.9 |
    | 80 KiB | 4 | 11.0 ± 0.9 |
    | 640 KiB | 4 | 60.2 |
    | 640 KiB | 8 | 58.1 |
    | 640 KiB | 16 | 59.5 |
- **N43 (P3 review fixes).**
  - **ISA check: the flag store is found structurally (6.3.5, N34).** The old check took "the
    body's first 32-bit `SCOPE_SYS` store" to be the flag. After a scope downgrade of the flag's
    release, that match moved to `note_failure`'s ABORT-word store, which also has
    `global_wb scope:SCOPE_SYS` in front of it and the barrier and pushes above it. So the check
    passed on exactly the regression it exists to catch.
    - The check now anchors on the post-push barrier: the first `s_barrier_wait` after the first
      16-B push store. The first global/flat/buffer store or atomic after that barrier must be the
      flag: a 32-bit `SCOPE_SYS` store, immediately preceded by `global_wb scope:SCOPE_SYS`.
    - It also checks the acquire side. Between the flag store and the next barrier there must be a
      `SCOPE_SYS` poll load and a standalone `global_inv scope:SCOPE_SYS`, one not directly after a
      load. The `spin_acq = 1` acquire loads carry their own `global_inv`, which does not count.
    - Negative cases, run on edited copies of the emitted `.s`, all rejected:
      - flag `global_wb` and store both downgraded to `SCOPE_DEV` (the reviewer's case, which the
        old check passed);
      - the flag store alone downgraded to `SCOPE_DEV`;
      - the writeback removed;
      - the writeback downgraded to `SCOPE_DEV`;
      - `.amdhsa_workgroup_processor_mode 0`;
      - the acquire fence's `global_inv` downgraded to `SCOPE_DEV`.

      The unedited ISA passes, with the flag at body line 278.
    - The check still reads a separate `-S` compile, not the linked object. A reviewer
      disassembled the linked `.hip_fatbin` and found the same code today. `src/kernels/
      CMakeLists.txt` now says both commands must expand exactly `R4DX_KERNELS_FLAGS`, with no
      per-command device flags.
  - **`test_tp_allreduce_cpu_peer` uses the 500 ms timeout (N39).** It used
    `kArTimeoutMaxMs` = 1500 ms, which is the warm-up value of 2.9 step 9, not a test setting, and
    exceeds P3's 500 ms spin bound.
  - **Pinned D2H targets in `HostMailboxComm`.** `SelfTest` read its result into a pageable
    `std::vector` with `hipMemcpyAsync`. HIP performs a copy to unpinned memory synchronously
    (`hip_runtime_api.h`), so the rank thread blocked in the driver behind the spinning kernel,
    and the 30 s `SyncWithWatchdog` after it could never fire. `SelfTest`'s buffer, and the
    `SeqAdvance` and `ReadStatus` targets, are now `core::PinnedBuffer`s.
  - **Recovery after a DEVICE timeout, on real GPUs (`test_tp_allreduce_2gpu` case 5).** Before
    this, only recovery after a host `Abort()` was exercised. The new case arms fault injection
    kind 1 (`kFaultStall`) on rank 1 at its 50th all-reduce and runs 200 10 KiB all-reduces
    through `RunDriver`. Then it checks:
    - rank 0's ABORT word is `kAbortTimeout` and rank 1's is 0;
    - rank 0's Status is claimed, with code timeout, channel 0, phase flag-wait, sticky set, and
      skipped blocks > 0;
    - rank 1's Status is unclaimed, with blocks bailed on the abort word;
    - both `CheckHealthy()` calls throw `TpAbortedError`, and rank 0's says "timeout at channel 0";
    - `Recover()` gives `new base == old base + max(SeqAdvance) + 64`;
    - 10,000 clean all-reduces follow, and `CallCounts()` agree with each other and with rank 0's
      `Stats().ar_calls`.

    Case 4 now checks its base the same way.
  - **Measured in that case: HIP here defers submitting launches.** The call that timed out was
    rank 0's FIRST of the run, not its 50th. The claimed seq was the first call's, all 150 of rank
    0's outputs were its own unreduced partials, and rank 1's first call was the only one of its
    150 that reduced correctly.
    - So rank 1's first 49 launches, enqueued before its 700 ms host sleep, did not run until after
      it. That is consistent with the runtime holding queued launches until a later API call
      flushes them.
    - Rule for P2b/P4: a rank's all-reduce kernels may start only when its queue is next flushed
      (an event query, a sync, or more launches), not when they are enqueued. Host work between
      enqueueing a token and the next flush shows up as the peer's spin, bounded by the 500 ms
      timeout.
  - **`HostMailboxComm::ResetCounters` also zeroes the `Stats()` deltas** (2.5 step 6):
    per-channel calls and bytes, host exchanges and their max wait. `Stats().aborts` stays
    cumulative over the endpoint's life. Before, `Stats().ar_calls` kept counting across
    recoveries while `CallCounts()` restarted.
  - **Streams are never evicted (N36).** `TrackStream` silently dropped the oldest of 8 tracked
    streams, and recovery step 1 then no longer waited on it. A ninth distinct stream now throws
    `std::logic_error` before the launch.
  - **A stuck endpoint leaks its own memory too.** If its stream is still busy after the
    destructor's 30 s watchdog, or the group is already marked stuck, `~HostMailboxComm` now
    leaks its VRAM, pinned buffers and stream, like the mailbox (2.6), instead of freeing memory a
    kernel or a queued D2H may still touch.
  - **`tool_tp_ar_stress` exit codes.**
    - A host error on any rank is reported first: exit 1, status "error".
    - "timeout" (exit 3) now means a device-claimed `kAbortTimeout`. Before, any rank host error
      also left an abort message, via `RunDriver`'s `Abort(kAbortHost)`, and was reported as a
      timeout.
    - Any other abort is exit 3 with status "aborted".
  - **`TpSetup` (tests/kernels/tp_ar_harness.h) has a destructor.** When a test or tool unwinds
    past `Destroy()`, it destroys the endpoints and streams on their rank threads through the
    `RankPool` it was created on. It leaks them and the group if that fails. Before, they were
    destroyed on the main thread during unwinding, on the wrong current device.
  - **Per-rank errors in the tools.** `tool_tp_ar_stress` and `tool_tp_ar_latency` print each
    rank's own exception to stderr, with its HIP device, from `RankPool::on_error`. `RunAll`
    rethrows only one root cause, and the G5 run in N44 lost the other rank's error that way.
- **N44 (P3 G5 gates, measured 2026-09-24 after the N43 fixes, both GPUs, production server
  stopped).** Rank 0 was device 1 (headless) and rank 1 was device 0 (desktop live).
  - **ISA check:** passes. The flag store is at body line 278, the poll at 301 and the acquire
    fence at 389.
  - **`test_tp_allreduce_cpu_peer`** (device 1, 500 ms timeout) passes in 3.0-3.5 s: 100,000 and
    wrap bit-exact; the timeout case took 100.08 ms on the device.
  - **`run_tests.ps1 -TwoGpu`** passes in 27 s:
    - 1,000,000 mixed all-reduces in 25.4 s;
    - `Abort()` -> exit in 0.076 ms;
    - host-abort recovery `0x00001000 -> 0x000b8221`;
    - device-timeout case: rank 0 timed out at channel 0, seq 761758 (its first call, N43), with
      596 blocks skipped; rank 1 had 4 abort exits and 592 skips; recovery
      `0x000b8221 -> 0x000ba073` (+7698 + 64);
    - 10,000 clean all-reduces after each recovery.
  - **`tool_tp_ar_stress --count 10000000 --pattern isolated`:** 10,000,000 10 KiB all-reduces
    verified in 66.0 s (151k/s; tp_bench about 146k/s).
  - **`tool_tp_ar_stress --count 10000000 --pattern decode` (the G5 line) did NOT complete: a
    Windows TDR, not the protocol.**
    - 1,417,088 decode-pattern all-reduces verified bit-exact at 7.26k/s, with 0 mismatches and
      0 timeouts. Then, after about 200 s, `hipEventQuery` returned HIP error 719 (unspecified
      launch failure). That is more than tp_bench's whole decode-pattern stress (1.28M).
    - Windows Error Reporting logged LiveKernelEvent **141** (`VIDEO_ENGINE_TIMEOUT_DETECTED`, a
      TDR) at 18:52:18, with dump `WATCHDOG-20260924-1852.dmp`. Two seconds later a desktop app
      crashed inside AMD's user-mode D3D driver, so device 0's adapter was reset.
    - Another 141 in the same WER bucket, at 18:49:44 (52 s into the run), did not stop it. The
      same bucket also appears at 00:23 today and on 2026-09-19 during tp_bench's development.
    - No TDR occurred in the 10M isolated run, the `-TwoGpu` test (2 MiB fillers) or the latency
      runs below.
    - Likely cause (not proven, and the dumps are not readable without admin rights): the decode
      pattern keeps device 0, which has `ComputePreemptionSupported=0`, saturated with back-to-back
      full-occupancy 60 MiB fillers. At 7.26k/s the 10M line is 23 minutes of that, and the
      desktop's own GPU work times out. The 500 ms spin bound (R6) does not address that.
    - Not retried: two TDRs in 200 s, and WER holds two earlier `VIDEO_TDR_FAILURE` (116)
      bugchecks on this box.
    - For P4 (G8, 10.x): on this box these TDRs appear **only** as WER LiveKernelEvent 141 in the
      Application log (provider "Windows Error Reporting", `P1: 141`), with no System event 4101.
      A check for 4101 alone would have missed both.
  - **`tool_tp_ar_latency`** (decode pattern, 50 tokens, 60 MiB / 4 MiB fillers), max over ranks:

    | Size | nb | `L_vs_no_ar_kernel` (us) | `L_vs_standin` (us) | G5 limit (us) | tp_bench (us) |
    |---|---|---|---|---|---|
    | 10 KiB | 4 | **5.58 ± 0.83** | 4.37 ± 0.77 | 9.0 | 7.47 ± 0.38 |
    | 80 KiB | 4 | **12.54 ± 0.55** | 10.20 ± 0.54 | 16.0 | 13.33 ± 0.38 |
    | 640 KiB | 4 | 61.94 ± 1.17 | 47.85 ± 0.86 | none | none |
    | 640 KiB | 8 | 59.10 ± 0.66 | 51.00 ± 0.47 | none | none |
    | 640 KiB | 16 | 59.30 ± 1.08 | 53.98 ± 0.85 | none | none |

    Per-token ms (all-reduce / stand-in / fillers only): 15.244 / 14.685 / 14.531 at 10 KiB and
    16.231 / 14.925 / 14.626 at 80 KiB. The best 640 KiB `nb` is 8, only 4.6% better than nb 4
    (below 11's 10% rule), so `--tp-ar-nb-large` stays 4 and no nb-8 stress is needed.
  - **Isolated-style latency** (the same tool with `--filler-mb 0 --dirty-mb 0`, so condition (c)
    is empty kernels), `L_vs_no_ar_kernel`:
    - 10 KiB: 4.51 ± 0.10 us;
    - 80 KiB: 11.02 ± 0.13 us;
    - 640 KiB: 59.06 ± 0.10 us.

    tp_bench's isolated us/AR at nb 4 are 5.34 / 11.7 / 58.3.
  - **`run_tests.ps1`** (default, `-LE tp2gpu`, device 1): 71 of 72 pass, including the CPU-peer
    test. The one failure is the known `test_mtp CheckSampledRoundsMatchPlain [w4a16]` (N29).
  - **G2** (`tp1_identity.ps1`, device 1, `-Image` = the golden vision image): PASS. All 12 rows
    are byte-identical to the baseline: 1-5, `row6_{bf16,w4a16,w4a8,mxfp4}`, 7 (vision), 8 and 9.
  - **G5 verdict:** the latency limits, the 640 KiB `nb` record, the ISA check and G2 are met.
    The "10M, 0 mismatches / 0 timeouts" line is not: the decode-pattern run ended in a device-0
    TDR at 1.42M, with 0 mismatches and 0 timeouts up to then. The protocol's 10M evidence is the
    isolated run. How to get 10M in the decode pattern without a TDR on the desktop card is left
    to the user: in chunks with idle gaps, with the desktop idle, or with the display moved off
    device 0 (Appendix C, question 2).

**P2b**

- **N45 (6.5, 6.6, 2.4: `EmulatedComm`, `tp/tp_comm_emulated.cpp`).**
  - **The all-reduce barrier waits at most the all-reduce timeout**, not the exchange's 30 s:
    `HostExchange` gains `AllGatherFor(..., timeout)` / `BarrierFor(rank, timeout)`; `AllGather` and
    `Barrier` keep the constructor's 30 s. A barrier that times out is reported the way the device
    kernel reports its spin timeout (6.3.5): the rank stores `kAbortTimeout` in its abort word,
    claims its host-side `Status` (channel, `seq` = the call index, phase flag-wait), the exchange
    is already poisoned by the timed-out wait, and the call throws `TpAbortedError("tp: rank <r>
    timeout at channel <c> block 0 seq <n> phase flag-wait (emulated all-reduce waited <ms> ms for
    rank <p>); ...")`. With the 30 s bound, fault kind 1 (a 700 ms stall on the peer) could never
    time out under emulation, and 10.1's asymmetric-fault case would test nothing. Warm-up's 1500 ms
    applies through `SetAllReduceTimeoutMs`, as for the real transport.
  - **Emulate mode has no mailbox.** `TpGroup` holds two host abort words (each written only by its
    own rank, like `ABORT[r]`) and each rank's two exchange-buffer pointers (published at endpoint
    creation, read by the peer's add). `ResetMailbox` clears the abort words (recovery step 4);
    `AllocateMailbox` refuses emulate mode; `SeqAdvance` returns 0 (no device seq counters), so
    recovery moves the base by 64 only. `Create(kEmulate, ...)` needs world 2, like `kReal`.
  - The stream wait before every emulated barrier records ONE endpoint-owned marker event and polls
    it (`SyncWithWatchdogEvent`, the event-reusing form of `SyncWithWatchdog`, N33): 30 s bound, no
    `hipStreamSynchronize`, no event created per call.
  - Invariant behind "no second barrier": when a rank passes barrier c + 1, its call-c add (the
    only reader of the peer's `xchg[peer][c & 1]`, which the peer rewrites at call c + 2) has
    finished. Draining call c + 1's stream proves that only when both calls share a stream, so a
    call on a different stream than the previous call's add also drains that add's stream first
    (N53).
  - `~EmulatedComm` syncs only its own stream: the tracked streams are the Model's, destroyed first
    (2.6). One rank's add reads the OTHER rank's exchange buffers, so `~TpModel` drains every rank's
    streams in a separate command before any rank frees anything (N47).
  - `SelfTest` is 6.3.9's pattern through the emulated all-reduce (80 all-reduces, so `CallCounts()`
    read `{48, 32}` right after a recovery).
- **N46 (2.7: `r4dx_topk_lse_f32_ws`).** Both kernels take the four partial arrays as pointers,
  and a null workspace makes the KERNEL select the module scratch -- instead of the host passing the
  globals' device addresses (`hipGetSymbolAddress`, per call and per device). `r4dx_topk_lse_f32` is
  `r4dx_topk_lse_f32_ws(..., 0)`: the same partials land in the same scratch, TP=1 outputs are
  unchanged. The workspace is the scratch's four arrays back to back -- ids 131,072 B, vals
  131,072 B, max 2,048 B, sum 4,096 B (8-aligned) = `r4dx_topk_lse_workspace_bytes()` = 268,288 B --
  and must be 16-byte aligned. A rank's workspace is allocated in `Model::Load` next to
  `argmax_pair_dev_` (every TP rank, bare or under `TpModel`), not in `TpWarmup`: no sampled path can
  reach a missing workspace, and nothing is allocated lazily inside a collective.
- **N47 (2.9, 2.2, 2.6: `TpModel::Load` and shutdown).**
  - Steps 3 and 5-10 run in every mode. Step 3's real-mode requirements (distinct PCI bus, same
    `gcnArchName`, `canMapHostMemory`) and step 4's wall-clock check stay with `--tp-mode real` (P4's
    table: "device validation (2.9 steps 3-4)"); the probe itself runs and logs in every mode.
  - Step 5 runs after step 6: the joint embedding decision needs `text.embed_tokens`' byte count,
    which the pinned load (step 6) yields; the free bytes are step 3's (before any rank allocated).
    The rule per physical device (PCI bus) is `free >= 2 * embed_bytes * ranks on it`.
  - After step 10 the facade drops its reference to the pinned table on rank 0's thread, so the last
    owner (a rank's `Container`) frees it on a rank thread, never on the facade.
  - noop: one rank thread (rank `noop_rank`) with `MakeNoopComm`. Its waits use
    `ProgressWatchdog::kNoStallLimit`: `NoopComm` never bumps a heartbeat, so the 60 s watchdog would
    kill any long command. Fault injection is refused in noop mode.
  - Every load-time command (probe, embed load, endpoints, `Model::Load`, the capability read) waits
    with `kNoStallLimit` (N8); the warm-up with the 60 s watchdog (its all-reduces bump heartbeats).
  - The warm-up latency sample (200 x 10 KiB, timed by the first rank thread) runs on a
    `RankSlot`-owned stream that lives as long as the endpoint, because the endpoint keeps every
    stream an all-reduce was queued on and syncs it in recovery.
  - Shutdown after the idle wait is two commands per rank: drain the endpoint's streams (all ranks
    first), then destroy the Model, the endpoint and that stream and `hipDeviceSynchronize()`; each
    wait is 30 s, then `quick_exit(3)`.
  - Each command's closure is held by a `shared_ptr`. (The first version let a closure a stalled
    rank was still running reference the caller's stack after the facade threw `TpTimeoutError`;
    N53 moved that state onto the heap and poisons the group on a stall.)
- **N48 (2.4, 2.8: the facade surface).**
  - Diagnostics and test hooks on `TpModel`: `GetState()`, `CallCounts()`, `CommStats()`,
    `ArmFaultInjection(rank, n, kind)` (counted from the call; `TpOptions::fault_*` calls it right
    after warm-up) and `RunCollectiveForTest(fn(Model&, rank))`, one collective command with the
    same state check, allocation guard, `CheckHealthy` and abort-on-error as a forward call
    (`VerifyWindow` has no `TextModel` method).
  - `DecodeStepMtp*` / `DecodeStepDflash*` check the cached `MtpEnabled()` / `DflashEnabled()`
    first and throw `std::runtime_error` without a command (so without `kNeedsRecovery`); both are
    false until P5. `EncodeImages` and the profiled methods: state check, then
    `TpUnsupportedError`, no command.
  - `LoadTextModel` refuses any non-default `TpOptions` field at world 1 (`std::invalid_argument`),
    the analogue of the CLI's "any `--tp-*` with `--tp 1`".
  - `Model::Reset()` also resets the arena. After every normal call that is a no-op; after an
    exception mid-layer it drops the dead call's scratch, so a recovered rerun starts at a fresh
    Model's arena offset (the byte-identical rerun of 10.1).
  - `model_types.h` also takes `ProfileEntry` (`StepProfile` holds a vector of it).
    `ImageRows::SetFilled(on_host, rows)` sets `on_host()` / `rows()`; `LocalTextModel` counts the
    rows from the grids.
- **N49 (P2b scope: the verify merges; H6/H7 stay in P5).** P2b's table puts `VerifyWindow` under
  TP (7.6) and H2-H5 here, and the MTP head (H6, 8.1) and the DFlash drafter (H7, 8.2), with their
  files, in P5. So P2b lifts the `TpUnsupportedError` of `DecodeStepSampled` (N15: only when it
  samples), `VerifyWindow`, `CommitVerifiedWindow` and `ReadVerifyLogitsRow`; `DecodeStepMtp*` and
  `DecodeStepDflash*` keep theirs. H2's fingerprint `kind` is `"VERI" << 32` OR'd with 1 (full
  logits out) and 2 (row summaries); H1 gains bit 8 (row summary). `Model::MergeShardResults` does
  H3's and H4's merges in ONE gather (`[pairs][summaries]` per rank, ids made global first).
  `test_tp_emulation` drives `VerifyWindow` through `RunCollectiveForTest` on ranks sized with
  `dflash_draft_k = 7` and no drafter.
- **N50 (10.1 `test_tp_emulation`: the w4a8 gate; measured 2026-09-24, device 1).** Max per-row rel
  L2 vs TP=1 on the 40- / 70-token scripts: bf16 6.5e-3 / 3.5e-3, w4a16 6.9e-3 / 5.4e-3, mxfp4
  1.5e-2 / 1.6e-2 -- inside 10.1's bounds -- but w4a8 7.7e-2 / 7.9e-2, over its 5e-2. Not a TP
  defect: measured against the TP=1 **bf16** run, TP=2 w4a8 is 9.1e-2 / 8.4e-2 away while TP=1
  w4a8 is 1.06e-1 / 1.03e-1 away -- TP is CLOSER to exact arithmetic (w4a16: 6.63e-2 vs 6.65e-2;
  mxfp4: 1.32e-1 vs 1.33e-1). The int8 per-row activation scales over the rank's local K (4.3, R10)
  change w4a8's quantization, and this 4-layer container's w4a8 logits are that sensitive. The w4a8
  gate is therefore: rel L2 vs TP=1 <= 1e-1 per row, AND max over the script of rel L2(TP=2 vs TP=1
  bf16) <= 1.0 x the same for TP=1 w4a8 (1.25 x until the P2b review tightened it, N53; measured
  ratios 0.853 / 0.822). bf16, w4a16 and mxfp4 keep 10.1's bounds; G4 (w4a16) is unaffected. This
  replaces a 10.1 bound; the user left the call to the implementer, so it stands (Appendix C,
  question 4). The rest of the run:
  - `DecodeStepGreedy` == argmax(`DecodeStep`) on 16/16 rows, every layout.
  - Sampled: 3 configs x 3 seeds per layout plus T = 0.005, all trajectories equal; 17-23 of 25
    rows fell back to the gathered full row at T = 1.0 and at top_p 0.95 / min_p 0.02 (H5
    exercised), 0 at top_k 20.
  - `VerifyWindow` (w4a16, 8 rows): ranks identical; merged preds == argmax of the gathered rows;
    merged top-64 == `r4dx_topk_lse_f32` over the gathered rows exactly; |lse delta| 0.
  - Faults: kind 0 reached the caller as "tp fault injection"; kind 1 as "tp: rank 0 timeout at
    channel 0 block 0 seq 252 ...". After each `Reset()`: `kReady`, `CallCounts()` {48, 32} on both
    ranks, rerun byte-identical to a fresh `TpModel`'s run.
  - `g_tp_collective_allocs` 0. Whole test 55 s.
- **N51 (9.1, 10.1: CLI and the teacher-forced tool).**
  - r4dx-cli's `--tp*` flags follow 9.1. In P2b the default `--tp-mode real` is itself refused
    (P4), so `--tp 2` needs `--tp-mode emulate` or `noop`. `/image` in `--chat` throws at that turn
    under `--tp 2` (the `--image` flag is refused at parse time). `--stats` at `--tp 2` prints the
    load line plus one VRAM line per rank, and the per-turn line's VRAM is the maximum over ranks;
    the `[stats] tp:` line is P4's. `tests/cli/test_args.cpp` gains `TestTpFlags`.
  - `tool_teacher_forced_logprobs`: `--tp`, `--tp-mode`, `--tp-devices`, `--tp-rank` (noop) and
    `--embed-device-resident`. The `[vram]` line sums `Vram()` over DISTINCT devices (emulate's two
    ranks share one device). The sidecar carries `"tp_world"` only when > 1, so TP=1 sidecars are
    unchanged. `test_teacher_forced_logprobs` drives the pass through `LocalTextModel`.
- **N52 (P2b smoke, measured 2026-09-24 on device 1, production server stopped; G2/G4 not run).**
  - CPU tests: 28 pass, `test_position_ids` skipped (no golden data). GPU on device 1:
    `test_tp_emulation`, `test_tp_loader`, `test_tp_allreduce_cpu_peer`, `test_topk_lse`,
    `test_summary_sampler`, `test_sampler_canonical`, `test_forward_smoke`,
    `test_teacher_forced_logprobs` and `test_core` pass.
  - r4dx-cli `--tp 2 --tp-mode emulate` on `l4-allmtp` (`--layers 4`): greedy, seeded sampled, a
    two-turn `--chat` (the prefix-mismatch `Reset()` path) and `--tp-mode noop --tp-rank 1` all run.
  - v6 w4a16, `--tp 2 --tp-mode emulate`, the standard prompt, 64 tokens: coherent ("Silicon
    threads weave, / Parallel light in the dark, / Pixels bloom anew." then the GPU explanation);
    load 6.9 s with the container in the page cache (25.8 s right after a rebuild), 19.96 GiB used on
    device 1, 25.9-26.0 tok/s decode (both ranks on one card, a host sync at every all-reduce);
    byte-identical text on a rerun. Seeded sampled (T 0.7, top_k 20, top_p 0.8, seed 1): coherent, 0/64
    fallback rows. The emulated isolated L(10 KiB) at warm-up reads 55-95 us.
  - `tool_teacher_forced_logprobs --tp 2 --tp-mode emulate` on the 4-layer container: 60 rows, max
    |logsumexp| 4.7e-7.

**P2b review**

- **N53 (2.2 step 6, 2.4, 2.6, 6.5, 10.1: the P2b review fixes).**
  - **A watchdog stall no longer leaves a rank on the facade's stack (2.2 step 6).** When the
    progress watchdog fired, the facade threw `TpTimeoutError` while the stalled rank might still
    run a closure that referenced the facade frame (`fn`, the result slots, `pos0`/`fallback0`) and
    the caller's arguments (`token_ids`, the rng copies, `walks`). When the stuck HIP call returned,
    that rank could read a freed token vector in `Model::Prefill`'s chunk loop, or write a result
    into a reused stack frame. The group was not aborted either, so the rank carried on with the
    forward. Now:
    - `Run`'s stall branch poisons every endpoint (`Abort(kAbortShutdown)`) before throwing, so a
      late rank throws at its next comm operation. The facade reads its own copy of each endpoint
      pointer (`RankSlot::facade_comm`, set after the endpoint-creation command, cleared before
      teardown), never `endpoint`/`noop` while a rank may write them. `~TpModel`'s 2.6 step 1 uses
      the same copy.
    - `RunCollective`, `RunAll` and `RunOne` keep the caller's `fn`, the per-rank result slots and
      rank 0's re-cached counters in a heap `CmdState` that every posted closure co-owns. The
      forward methods capture their arguments by value (a `Prefill` copies its token vector once)
      and keep the rng copies and walk lengths in `shared_ptr`s. `Vram`, `CallCounts`, `CommStats`,
      `ArmFaultInjection` and `Reset` do the same. `TpGroup::Recover`'s step closures co-own `eps`
      and `advance` and capture `new_base` by value, and `TpModel`'s `RankRunner` copies them into
      the commands.
    - `TpModel::Load` declares every local that a load command touches before `m`. A failed load
      destroys `m` first, and `~TpModel` waits for every rank to go idle, or `quick_exit(3)`s,
      before those locals go.
  - **The pinned embedding table is never freed on the facade (2.1).** On a failed load (for
    example, both ranks' `Model::Load` throwing) the facade's `embed` could be the last owner and
    `hipHostFree` 2.37 GiB on a thread with no `hipSetDevice`. A guard destroyed before `m` now
    drops that reference on rank 0's thread. If rank 0 can no longer run a command (`kFatal`), the
    guard leaks the table.
  - **`EmulatedComm` no longer assumes one stream per rank (6.5, N45).** If a rank's all-reduces
    switched streams without a full sync, the peer's barrier at c + 1 no longer proved that this
    rank's call-c add had finished. The peer's call c + 2 could then overwrite the buffer that add
    was still reading, giving a silently wrong sum that both ranks would agree on. No current path
    switches streams, but nothing enforced it. A call on a stream other than the previous call's
    add now drains that add's stream too. One extra event wait per stream switch; recovery clears
    the flag.
  - **`--max-ctx` below 65 is refused by name under TP.** The warm-up prefills a 64-token chunk
    and decodes one token. Before, such a load died inside the warm-up with the KV cache's
    `start_pos+T exceeds max_context_tokens`.
  - **`Model::PrefillMultimodal` refuses `ImageSpan::embeds_on_host`** (`std::invalid_argument`)
    until P5 adds the H2D splice. Before, it would have run a D2D copy from a host pointer. At TP=1
    the flag is always false, so no byte changes.
  - **The w4a8 yardstick of `test_tp_emulation` is tightened from 1.25 x to 1.0 x (N50).** At 1.25 x
    it would pass an independent w4a8-only TP error of ~0.096 rel L2, as large as w4a8's whole
    quantization error. At 1.0 x the measured ratios, 0.853 and 0.822, pass, and an added error
    above ~5e-2 fails. The gate change stands (Appendix C, question 4).
  - **Test coverage added.**
    - `test_tp_emulation` now also checks that the injection policy set in `kNeedsRecovery` (false)
      is what both ranks run with after `Reset()`.
    - It adds a lockstep divergence: the ranks feed different tokens to one `DecodeStep`. That gives
      H1 `TpDivergenceError` on both, then `kNeedsRecovery`, then `Reset()` recovers, and the rerun
      equals the fresh run.
    - It adds a failed recovery: rank 1 armed to throw at its 5th all-reduce, then a divergence,
      which issues no all-reduce, then `Reset()`, whose `SelfTest` reaches the fault. The group
      goes `kFatal`, `Reset()`, `DecodeStep()` and `CallCounts()` throw `TpStateError("tp: fatal
      ...")`, and the cached accessors and `Vram()` still work. `~TpModel` then tears down cleanly.
    - `test_tp_host_exchange` adds `TestPerCallBound`: `BarrierFor(150 ms)` on a 30 s exchange
      times out at 150 ms and aborts the group, and `AllGatherFor` gathers normally after `Reset()`.
- **N54 (P2b gates, measured 2026-09-24 after the N53 fixes, HIP device 1 only, production server
  stopped).**
  - **CPU tests:** `test_tp_{config,shard,vocab_merge,host_exchange,rank_worker}` and
    `test_cli_args` pass.
  - **`test_tp_emulation`** passes in 54.6 s, with every N53 case included. The per-row max rel L2
    vs TP=1 is unchanged from N50:
    - bf16: 6.5e-3 / 3.5e-3;
    - w4a16: 6.9e-3 / 5.4e-3;
    - w4a8: 7.7e-2 / 7.9e-2, with yardstick ratios 0.853 / 0.822 against the 1.0 limit;
    - mxfp4: 1.5e-2 / 1.6e-2.
  - **`run_tests.ps1`** (default, `-LE tp2gpu`): 72 of 73 pass (15 of them skips), in 574 s,
    `test_tp_emulation` included. The one failure is the known `test_mtp`
    `CheckSampledRoundsMatchPlain [w4a16]` (0/18 identical sampled trajectories, N29).
  - **G4: PASS.** These are 10.4's commands with two differences.
    - `--ref-dir` is `C:\Users\pay20\dev\r4dx-m8\tools\reference\kl_out\ref`, the Rung 4 bf16
      reference dumps, only read. The tp2 worktree has no gitignored `kl_out\ref`. Its
      `kl_corpus\tokens.json` has the same SHA-256, and `kl_report.py` checks each segment's
      token-id hash.
    - `tool_teacher_forced_logprobs` does not create `--out-dir`, so the two directories were made
      first. `kl_report.py` ran under `C:\Users\pay20\dev\.venv` (numpy 2.4.3); the README's venv
      does not exist on this box.

    `kl_v6_tp2emu.json`, KL(ref || TP=2 emulated):

    | Segment | Mean KL | Top-1 | KL>1 |
    |---|--:|--:|--:|
    | cpp_source | 0.02397 | 94.43% | 0 |
    | english_prose | 0.02840 | 91.20% | 0 |
    | python_source | 0.02748 | 94.13% | 0 |
    | thai_prose | 0.07428 | 84.26% | 2 |
    | **ALL** (4092 rows) | **0.03853** (limit 0.0435) | **91.01%** (limit 90.43%) | 2 |

    - TP=1 from the same binary and run (`kl_v6_tp1.json`, an extra `kl_report.py` call): 0.03856 /
      90.86%. Milestone 11's 0.03851 / 90.93% was measured on 2026-09-22, before `main`'s `32093f2`
      w4a16 re-sweep (2026-09-24) changed the TP=1 tunings. G4's limits come from Milestone 11 and
      are met by both runs.
    - `kl_tp1_vs_tp2emu.json`: mean KL 0.00103 (10.4 expects ~1e-3), top-1 98.31%, max 0.0516, no
      position above 1 nat.
    - Wall time and VRAM: TP=1 took 121.5 s (29.7 ms/row) with a 17.07 GiB peak. Emulated TP=2 took
      154.0 s (37.6 ms/row) with 20.02 GiB (both ranks on one device). The largest |logsumexp| was
      1.9e-6.
  - **G2 (`tp1_identity.ps1`, `-Image` = the golden vision image): PASS.** All 12 rows are
    byte-identical to `build\baseline` (1-5, `row6_{bf16,w4a16,w4a8,mxfp4}`, 7, 8 and 9), in 437 s.
  - **Smoke (P2b gates' CLI line):** the text is coherent and byte-identical to N52's ("Silicon
    threads weave, / Parallel light in the dark, / Pixels bloom anew." then the GPU explanation).
    Load took 5.9 s, decode ran at 28.7 tok/s, and VRAM used was 19.96 GiB. The emulated isolated
    L(10 KiB) was 42.4 us.

## Appendix C -- Open questions for the user

1. **`PrefixState::Invalidate` bug on `main` (8.4).** After any exception inside a request,
   today's server skips `Reset()` on the next request and prefills on top of the failed request's
   state -- silent wrong output at TP=1. The fix is ~10 lines plus a CPU test. This design lands
   it on `tp2` in P5. Do you want it fixed on `main` now, independently of TP?
   **Resolved:** fixed on `main` in `977160e` (`needs_reset_`, lifted only by `Commit()`); `tp2`
   takes it when it merges `main`, so P5 no longer carries it.
2. **Device 0 for two-GPU tests and gates.** Every P3-P5 two-GPU run needs the production server
   on device 1 stopped (9.2) and puts load on the desktop card. Is there a preferred window, and
   should the desktop be kept idle (or the display moved off device 0, if this machine allows it)
   during P4's 60-minute soak (R2)?
   **Answered (2026-09-24): the display stays on device 0.** P4 keeps every GPU submission short
   instead: a forced flush after each prefill chunk and a cap on the work queued ahead of the host
   (decode already syncs every token). The soak is staged, 5 minutes before 60, with a TDR check
   (WER LiveKernelEvent 141 and System 4101) after each stage; the first TDR stops device-0 work.
3. **Stop rules.** If G3 (P2a) or G5 (P3) misses, the plan stops before P2b. If P4 lands below
   1.40x it stops; between 1.40x and 1.45x it pauses for your call. Confirm those thresholds, or
   name the speedup below which TP is not worth merging.
4. **The w4a8 numerics gate of `test_tp_emulation` (Appendix B N50, N53).** 10.1 bounds w4a8 at
   rel L2 <= 5e-2 per row vs TP=1; on the 4-layer container w4a8 measures 7.7e-2 / 7.9e-2, because
   the int8 activation scales of the row-parallel layers are taken over the rank's local K (4.3) --
   and that moves TP=2 CLOSER to the bf16 run than TP=1 w4a8 is (ratio 0.853 / 0.822). The test
   now gates w4a8 at <= 1e-1 per row vs TP=1 AND TP=2's distance from TP=1 bf16 <= 1.0 x TP=1
   w4a8's. Do you accept that replacement, or should w4a8 keep 10.1's 5e-2 (which fails today by
   design, not by a defect)?
   **Answered (2026-09-24):** the user left it to the implementer; the replacement stands.
