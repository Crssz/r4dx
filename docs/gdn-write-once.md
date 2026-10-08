# Write-once GDN state for speculative verify

Status: **on by default** since 2026-10-08 (`R4DX_GDN_WRITE_ONCE=0` or `R4DX_DECODE_LEGACY=gdnwo` restores the
window-slot path, every byte of every path as it was).

Measured on device 1 (E:\models\r4dx\gdnwo, E:\models\r4dx\round2; mix4.5m, `bench_decode.ps1`, 4 prompts):
every text SHA-256 identical to main and to the switch-off run in plain, `--dflash` k=7 and `--mtp 3`, at TP = 1
and TP = 2; tok/round and acceptance identical; `tp1_identity`, `kl_rung4`, `validate_dflash`,
`validate_spec_sampling -Quick` and `validate_fusion` pass with the switch on and off.

| | round ms off -> on | tok/s main -> default on | VRAM freed |
|---|---|---|---|
| plain | -- | 37.46 -> 37.40 (noise) | 0 |
| DFlash k=7 | 34.85 -> 33.54 (-1.31) | 119.25 -> 123.39 (+3.5%) | 0.92 GiB |
| MTP-3 | 34.15 -> 33.64 (-0.50) | 79.42 -> 80.56 (+1.4%) | 0.42 GiB |
| TP = 2 DFlash k=7 | 21.92 -> 21.15 (-0.77) | | 0.38 GiB per rank |

## What it does

The legacy recurrent kernel (`r4d_gdn_recurrent_update_k128_v128_bf16_fp32state`) stores a full fp32 state for
EVERY candidate row of a verify window, because which row survives is unknown until the whole forward pass is
over: 8 x 151 MB of stores per DFlash k = 7 round at TP=1, and a window-slot bank of 9 x 144 MiB resident. The
write-once path stores per row only what defines the row's state update -- `eg = e^g`, the correction `u` (one
per v row) and the normalised key `kk` (0.26 MiB per layer for 8 rows instead of 24 MiB) -- keeps ONE state `B`
per sequence, and applies the accepted prefix of the log inside the NEXT call's seed load, in registers, with the
legacy loop's own two roundings per element:

    h <- h * eg_t        (rounded)
    h <- h + u_t * kk_t  (rounded; a separate multiply and add)

State traffic per round: 1 read + 1 in-place write instead of 1 read + 8 writes. Plain T = 1 decode with nothing
pending runs the unmodified legacy kernel. Timing gate (already passed, branch `gdnprobe`, probe skipping the 7
stores): DFlash k = 7 round 35.1 -> 33.8 ms (verify GPU span 30.42 -> 29.06), MTP-3 34.4 -> 33.6, plain
unchanged; the real thing pays the log and the replay on top of that, estimated 0.1 to 0.25 ms.

## Correction to the first design: no `__fmaf_rn`

The design draft pinned `h += u * kk` with `__fmaf_rn`, assuming the legacy loop was fused by clang's default
`-ffp-contract=fast`. It is not: `third_party/CMakeLists.txt` builds every libr4d unit with
`-ffp-contract=off`, and the legacy kernels' ISA has 5 to 8 `v_fma` each (softplus / exp), none in the state
loops -- the update is a `v_mul_f32` and a `v_add_f32`. An fma replay would change committed states by up to an
ulp (`test_gdn_write_once_cpu`'s negative control: 7628 of 19200 elements differ). So there is **no refactor of
the legacy kernel at all** (the design's step 0): the replay is written in the same two-operation form,
`#pragma clang fp contract(off)` pins it (and everything in the new section of the unit) whatever the build
flags say, and the legacy section of the file is untouched. Proof of step 0: all 8 legacy instantiations
compile to identical ISA text (labels normalised) and identical metadata (VGPRs 229 / 243, SGPRs, kernarg
size, no spills) before and after. New kernels: 16 `r4d_gdn_recurrent_update_wo_kernel` instantiations
(ABF16 x NORM x VS x LOG/DIRECT), 231 to 238 VGPRs, no scratch, no `v_fma` in the state arithmetic; the replay
kernel 135 VGPRs (VOPD dual mul/add only).

## Pieces

* `third_party/libr4d/r4d_gdn_recurrent_update_k128_v128_bf16_fp32state.hip` (new section at the end),
  `r4d.h`, `r4d_registry.hip` (ops `gdn_recurrent_update_wo`, `gdn_state_replay`), `r4dx/core/r4d.hpp`:
  * `r4d_gdn_recurrent_update_wo_k128_v128_bf16_fp32state(..., log_in, log_out, pending, log_depth, mode, ...)`.
    N must be 1. `state` + `sidx[0]` address B (a slot <= 0 is skipped, as in the legacy kernel, so
    `GdnControlCache::SidxBase` needs no new key and TP prewarm is unchanged). `R4D_GDN_WO_LOG`: the call's rows go
    to `log_out`; no state store except the replay's in-place write-back of `pending` rows into B (only when
    `pending` is non-null: the first round after a prefill writes strictly less). `R4D_GDN_WO_DIRECT`: no log, the
    final state is stored into B after the last row (T = 1 with a pending prefix). Outputs `o` are the legacy
    kernel's bit for bit: the register chain is the same text.
  * `r4d_gdn_state_replay_k128_v128_fp32`: `dst = src + first n logged rows`, one workgroup per value head, in
    place or into a scratch slot (Materialize, the digest flush, the tests).
  * `r4d_gdn_wo_log_bytes(H, Hg, depth)`: log layout `u [depth][H][V] | kk [depth][Hg][K] | eg [depth][H]`, fp32,
    256-byte padded (263,680 B at W = 8, H = 48). Two buffers per layer, ping-pong: the call that replays the
    previous log writes the other half (other workgroups may still be replaying the old one; kk is shared across
    rows, value heads and the VS workgroups).
* `src/model/gdn_state.h`: `GdnStateManager(..., write_once)`: `slots_per_seq = 1` (`recurrent_` = B + the null
  slot), logs allocated in the constructor (no lazy `hipMalloc`, TP), `LogIn / LogOut / FlipParity`,
  `Materialize` (replay in place), `MaterializeTo` (scratch), `ShiftConv` (the conv half of `CollapseWindow`,
  unchanged), `WindowSlot` throws. With one slot per sequence `ConvSeqPtr`'s `SlotForSeq` indexing agrees with the
  `(max_seqs + 1)`-line conv buffer at any `max_seqs` (the window-slot path only at 1).
* `src/model/gdn_layer.cpp`: T == 1 and nothing pending: the legacy kernel in place (no `num_accepted`); T == 1
  pending: DIRECT; T > 1: LOG (+ replay when pending), then `FlipParity`. `GdnLayerParams::gdn_pending`. The conv
  kernel keeps `num_accepted` exactly as before.
* `src/model/gdn_write_once.h` (header-only, HIP-free): the switch parser and `DecideGdnWriteOnce`,
  `GdnSlotPlan`, `GdnLogRing`, `GdnLogFloats`, and `GdnPendingBook`, the whole rollback story:

| event | pending after |
|---|---|
| Prefill / Reset / RestoreCheckpoint / digest flush | false (a pending prefix is materialised first) |
| plain decode, T == 1 (`RunChunk`) | false (the kernel replayed it and wrote B) |
| any verify call launched | false (it replayed the previous log into B; its own rows are not committed) |
| `CommitVerifiedWindow(n)` after a verify with T > 1 | true, count n (= `mtp_num_accepted_dev_`, the existing acceptance thread) |
| the same commit after a T == 1 verify | false (the T == 1 kernel stored the state directly) |

  Rejecting rows is "do not mark them pending"; nothing is rolled back or copied.
* `src/model/model.{h,cpp}`: `ModelOptions::gdn_write_once`, `gdn_book_`, `FlushGdnPending`,
  `CollapseSpeculativeWindow` (write-once: flush B + the same conv shift under the same condition),
  `CommitVerifiedWindow` refuses `n` greater than the last verify call's rows (write-once only: the window-slot
  path would read a slot that call never wrote), `GdnStateBytes()`, `GdnWriteOnceEnabled()`; under
  `R4DX_TP_TESTING`: `DebugStateDigest` (flushes first, digests B only, never the logs) and `DebugLiveGdnDigest`
  (`gdn.live.<layer>`: the logical state in BOTH modes, so they compare bit for bit).

## Switches

| switch | effect |
|---|---|
| `R4DX_GDN_WRITE_ONCE=1` (or `on`) | write-once state, where the Model has a speculative window (window > 1). Unset / `0` / `off`: the window-slot path. `gdn_write_once.h`'s `kGdnWriteOnceDefault` is the one line that flips the default after the GPU gates pass |
| `R4DX_DECODE_LEGACY=gdnwo` (also `all`) | kill switch: window-slot path even when `R4DX_GDN_WRITE_ONCE=1`, where `ModelOptions::gdn_write_once` follows the environment. The switch that stays meaningful after the default flips |
| `ModelOptions::gdn_write_once` 0 / 1 | forces the choice whatever the environment says (the A/B tests load one Model of each in one process); -1 follows the environment |

A write-once Model prints one line at load; a window-1 Model ignores the request (nothing to save).

## Known limits

* A T == 1 verify mutates B in place (it is the plain-decode kernel), so two bare T == 1 `VerifyWindow` calls
  with no commit between advance B twice. Production always commits. For T > 1 the second bare call seeds its
  recurrent state from the committed state: the latent window-slot bug (a second bare verify after a round that
  committed n > 1 re-reads a window slot the first call overwrote) does not exist in the write-once path, pinned
  in `test_gdn_write_once_cpu` (both behaviours, recurrent state only) and `test_gdn_write_once_model` (the
  committed GDN state is unchanged by one and by two bare verifies). The window-slot path keeps the bug on
  purpose: it is the A/B reference and a fix would need a spare slot.
* A second bare verify's ROWS still differ from the first's in both modes: the conv history is a rolling buffer
  every verify call rewrites shifted by its own tokens, and the next call reads it at offset `num_accepted - 1`,
  i.e. inside the first call's inputs. Making a bare verify fully idempotent would need a conv double buffer.
* `DecodeStepProfiled` consumes a pending prefix like a plain step; `PrefillProfiled` never collapsed (as before).
* The plain `gdn.rec.*` digests of the two modes differ by construction (different allocation); compare
  `gdn.live.*`.
* The write-once kernel is N == 1 (the model's whole scope); a second sequence needs a per-sequence log offset.
* Slot 0 stays a dead 144 MiB (NULL_BLOCK_ID convention).

## Tests

CPU, run in this session (`HIP_VISIBLE_DEVICES=-1`): `test_gdn_write_once_cpu` (policy table, slot / ring / log
arithmetic against libr4d's own log size, the fp32 emulation of the kernel arithmetic: every (T, n) for T in
{1, 2, 5, 8}, 400 random scripts of rounds / plain steps / flushes with the real `GdnPendingBook` -- outputs and
logical state bitwise equal to the window-slot emulation after every event, flushes invisible -- the double bare
verify, the commit bounds, the fma negative control) and `test_decode_legacy` (the `gdnwo` token).

GPU, device 1, written and built, NOT run:

* `test_gdn_write_once`: the new kernels against the legacy kernel at the real rank shapes (H/Hg 48/16 and
  24/8, fused norm) and the no-norm VS = 2 / VS = 4 paths with fp32 a/b: `o` bitwise for T in {1, 2, 5, 8}; the
  replay of n logged rows equals legacy window slot n-1 for every n; no pending prefix means B untouched; scripted
  and random (T, n) chains with plain steps (direct path with and without a pending prefix), full-window commits,
  commit 1, the final in-place Materialize; refusals; a NULL_BLOCK_ID slot is skipped.
* `test_gdn_write_once_model`: two Models (and two TpModels, emulate) with `gdn_write_once` 0 and 1 on the
  4-layer MTP container, bf16 and w4a16: verify + commit scripts over many (T, n), plain steps, extends (after a
  multi-token round, after an n = 1 round of T > 1, after a plain step), checkpoint Save / Restore, Reset, real
  MTP k = 3 rounds, the double bare verify, the refused over-commit, window 1; logits, tokens, live GDN / conv /
  KV digests bitwise equal after every event, with a digest at the end only and after every event.

## GPU steps (device 1, one job at a time)

```powershell
cd C:\Users\pay20\dev\r4dx-gdnwo
$env:HIP_VISIBLE_DEVICES='1'
.\build\win-hip\tests\model\test_gdn_write_once.exe          # kernel level: pass = exit 0, "test_gdn_write_once: OK"
.\build\win-hip\tests\model\test_gdn_write_once_model.exe    # model level: pass = exit 0, "[PASS] test_gdn_write_once_model"
# regressions of what the change touches with the switch OFF (must be as on main):
.\build\win-hip\tests\model\test_mtp.exe; .\build\win-hip\tests\model\test_prompt_checkpoint.exe
.\build\win-hip\tests\model\test_gdn_layer.exe; .\build\win-hip\tests\model\test_forward_smoke.exe
# the A/B bench (text stable column, tok/s, VRAM): the same container twice, the second with the switch on
.\tools\quant2\bench_decode.ps1 -Container 'off=<container>' 'wo=<container>^R4DX_GDN_WRITE_ONCE=1' -Modes plain,dflash7,mtp3
# TP = 2 (real): add 'off2=<container>!2' 'wo2=<container>!2^R4DX_GDN_WRITE_ONCE=1'
```

Expected: `text stable` True and identical `text_sha256` between `off` and `wo` in every mode; plain tok/s within
noise; DFlash k = 7 and MTP-3 faster by about 1.3 and 0.8 ms per round (the probe's 1.35 / 0.8 minus the log and
replay cost); VRAM lower by about 0.96 GiB per rank at TP = 1, k = 7 (about 0.48 GiB per rank at TP = 2).
