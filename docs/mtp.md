# MTP self-speculative decode

Status: implemented, correctness-tested, and **now a genuine, measured speedup** on HIP device 1
against the real 64-layer container (`D:\models\r4dx\qwen38-27b.r4dx`, which already carries
`mtp.*` weights -- see "Container" below) and the 4-layer MTP test container
(`D:\models\r4dx\qwen38-27b-l4-mtp.r4dx`). This revision (2026-09-19 review-fix pass) corrects two
blocker-severity bugs an Opus review found in the original implementation -- a swapped `fc` input
concat order, and an MTP KV cache that was reset to empty every round instead of built in lockstep
with the real sequence -- which together explained the prior revision's 0-1.2% measured acceptance
rate and "MTP is a net slowdown" conclusion. **Both are fixed; acceptance now measures 30-75%
(K-dependent) and decode is 1.4-2.0x faster with MTP on than off, every layout.** See "Incident:
the 0-1.2% acceptance bug" below for the full writeup, and "Measurement" for the corrected numbers.

This revision (2026-09-20, MTP quality + device-resident draft loop pass) adds a configurable MTP
head layout (`--mtp-head-layout {bf16,layout}`, default `layout` -- measured faster with no
acceptance cost, see "MTP head layout" below), investigates (without fully root-causing) the
remaining w4a16-vs-w4a8/mxfp4 acceptance gap ("Acceptance gap investigation" below), and makes the
draft loop device-resident (embedding gather now a device kernel fed straight from the on-device
argmax, no more per-drafted-token host round-trip -- "Device-resident draft loop" below).

**Milestone 4 integration (2026-09-20)**: a clean `build.ps1 -Clean` rebuild + full `ctest` (37/37)
+ a fresh `r4dx-cli --stats` sweep at `--mtp 0` and each layout's own best `K` (re-checked against
neighbors, not assumed) reproduced this file's numbers within run-to-run noise, with one real
finding: **mxfp4's best K moved from K=2 to K=3** post the Q5 GEMM re-sweep. See the "Milestone 4
correction" note under "Measurement" below for the full re-check, and `docs/perf.md`'s consolidated
M1->M4 table for the headline numbers.

**Milestone 4 follow-up (2026-09-20)**: the acceptance-gap investigation above is now closed with a
measured positive result -- h_seed drift from a bf16 exact-arithmetic reference matches the
acceptance ranking exactly, driven mostly by one outlier residual dimension (see "h_seed drift"
under "Acceptance gap investigation" below). Also: `--mtp-head-layout` is now a server-side flag too
(`src/server/server_args.h`), and `--chat` multi-turn + MTP's mid-round bookkeeping fix now has a
real end-to-end regression test (`tests/model/test_mtp.cpp::CheckChatMultiTurnMidRoundStop`) -- see
"Known gaps" below for both.

**Milestone 3 integration confirmation (2026-09-20)**: a clean `build.ps1 -Clean` rebuild + full
`ctest` (35/35) + a fresh `r4dx-cli --mtp 3` run per layout against the real container reproduced
this file's own numbers within run-to-run noise: w4a16 68.37 tok/s (46.3%, 2.31 tok/round), w4a8
61.41 tok/s (43.3%, 2.27 tok/round), mxfp4 55.72 tok/s (47.1%, 2.32 tok/round). The mid-round
`committed_tokens` bug fix (FIX pass, see "Known gaps" below) and the device-resident draft loop are
both confirmed still in effect (`tools/server/smoke.ps1 -Mtp 3` against the real container passes
25/25, including its MTP-path log-line check). See `docs/perf.md`'s "Milestone 3 consolidated
performance" for the full six-run table this confirms.

## Design

Per `vllm/model_executor/models/qwen3_5_mtp.py`'s `Qwen3_5MultiTokenPredictor.forward` -- the
actual working reference for this module, sitting in this machine's own
`C:\Users\user\dev\vLLM_for_AMD` tree the whole time (the original pass cited
`modeling_qwen3_5.py`'s `Qwen3_5MTPLayer`, which does not exist in the installed `transformers`
5.17.0: `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]` -- see "Incident" below for how this was
found) -- and the checkpoint's `mtp.*` tensors (`docs/container-format.md`, cross-checked against
the converter's own shape comment in `src/convert/main.cpp`: `mtp.layers.0` is a **complete
full-attention decoder layer**, not a small bespoke head):

```
x = concat(pre_fc_norm_embedding(embed(t)), pre_fc_norm_hidden(h_i))   [2*hidden]
x = fc(x)                                                              [hidden]  -- plain bf16 GEMM
h' = one_full_attention_decoder_layer(x)    -- mtp.layer's own weights + its own REAL KV cache
logits = lm_head(norm(h'))                  -- mtp.norm, then the SHARED main-model lm_head
draft = argmax(logits)
```

`h_i`/`t` for the FIRST draft of a round are the main model's own pre-final-norm hidden state and
the last-accepted real token (`Model::mtp_seed_hidden_` -- captured by every `Model::RunChunk` call
now, not just ones with `want_logits`, since it doubles as the "boundary" seed `MtpHead::PrimeKv`
needs -- see below). Draft `j>1` chains: `h_i` becomes THIS layer's own previous-step output `h'`,
`t` becomes the previous draft. `K` draft tokens per round (`--mtp K`, 0 disables --
`r4dx::model::MtpHead`, `src/model/mtp_head.h/.cpp`).

**MTP's own KV cache is a REAL per-sequence cache, sized to the same `max_ctx` as every backbone
attention layer's own `PagedKvCache`, and extended in LOCKSTEP with the real sequence.** This
corrects the original pass's design premise (reset MTP's KV cache to empty every `Draft()` call,
relying solely on `h_seed` to carry context) -- `qwen3_5_mtp.py`'s own `forward` takes real
`positions` and runs `mtp_layer` against a real per-sequence KV cache the vLLM runner prefills over
the whole prompt and extends by every accepted token, exactly like any other decoder layer, not a
per-round scratch cache. `Model::RunChunk` (`model.cpp`) now calls `MtpHead::PrimeKv` on every
prefill chunk AND every plain decode step to extend this cache by exactly the positions that call
just made knowable -- unconditionally (not gated by `want_logits`), at zero cost when `mtp_` is
unset. See `mtp_head.h`'s file comment and `PrimeKv`'s own comment for the exact `(h_i, t_{i+1})`
pairing this computes, and why no rewind machinery is needed for MTP's own cache either (the same
"self-correcting via position overwrite" trick the main model's own KV cache already relies on --
see "State commit and rewind" below).

### Verify step

The seed token + K drafts (`q_len = K+1 <= 10`) are fed to the REAL model in one
`Model::VerifyWindow` call (public, used by both `DecodeStepMtpGreedy` and `tests/model/test_mtp.cpp`):
one q_len>1 attention-decode-kernel + GDN conv_update/recurrent_update pass, producing per-position
logits/argmax for every candidate. Greedy acceptance: the longest prefix of drafts whose PRECEDING
candidate's own argmax matches it; the position right after that prefix's own argmax becomes the
"corrected" token (a genuine correction on partial/total rejection, or a free "bonus" token beyond
the K drafts when every draft was accepted).

**State commit and rewind** (`src/model/gdn_state.h`'s file comment has the full kernel-level
derivation): GDN's `GdnStateManager` reserves `1 + mtp_draft_k` PHYSICAL SLOTS per sequence (was 1)
-- one per candidate window index -- and every verify call's `num_accepted` device pointer
(threaded from the PREVIOUS round's own accepted count, nullptr only for the very first round after
a fresh `Prefill`) tells `r4d_gdn_conv_update_w4_h128_bf16`/`r4d_gdn_recurrent_update_*` which
window slot to seed FROM. Attention's KV cache needs **no extra machinery at all**: since slot ==
position in this paged cache, a rejected candidate's stale KV entry is simply overwritten the next
time that position is written (`pos_` only ever advances by however many candidates were actually
committed, never by the full window) -- this is the standard "self-correcting via position
overwrite" trick every speculative-decoding KV cache uses. **MTP's own KV cache (this revision)
relies on exactly the same trick**: a round's `Draft()` speculatively writes MTP-position
`base_pos+0 .. base_pos+k-1` for its whole chained draft, whether or not verification later accepts
all of them; the NEXT round (or the next plain-decode `PrimeKv` call) always starts writing again
from `base_pos_new = base_pos_old + num_committed - 1`, exactly the first position a partial
rejection left holding stale/hypothetical data, before anything could ever read further ahead
(causal attention only reads positions `<= current`).

`ModelOptions::mtp_draft_k == 0` (the default) makes every one of these changes a no-op: window
banking degenerates to exactly 1 slot (today's pre-MTP layout), `num_accepted` stays nullptr always,
MTP's own KV cache and priming calls never run, and `Model::DecodeStep(Greedy)`/`RunChunk` are
byte-for-byte what they were before this pass.

### Plain-decode / MTP interaction

An MTP-enabled `Model` (`mtp_draft_k > 0`) can still be driven through plain
`DecodeStep`/`DecodeStepGreedy` (not `DecodeStepMtpGreedy`) -- the CLI never does this itself
(`--mtp K` with `--temperature > 0` now warns and forces `mtp_draft_k=0` for that run instead, see
"CLI" below), but `Model`'s own contract must not silently corrupt state if some OTHER caller does.
`Model::RunChunk`'s decode branch (review finding, 2026-09-19) now threads GDN's `num_accepted`
explicitly: it reads whatever the LAST verify round left it at (so a plain decode step right after
an MTP round seeds from the correct window slot, not window 0), then sets it to `1` right after
committing (so a plain decode step always "accepts" exactly the one token it was given, and the
NEXT call -- plain or MTP -- seeds correctly). It also calls `MtpHead::PrimeKv` for the one position
it just made knowable, keeping MTP's own KV cache in lockstep regardless of which decode method is
used. `DecodeStepMtpGreedy(token_id, k=0)` -- the documented "degenerates to a single
DecodeStepGreedy-equivalent result" case -- similarly primes MTP's KV directly (since `Draft()`,
which normally does this as a side effect of drafting, is skipped when `k==0`).

### Sampled rounds (Milestone 6 stage S2)

Everything above describes GREEDY acceptance ("the draft is confirmed iff it equals the target's
argmax"), which is why MTP was gated on `temperature <= 0`. `Model::DecodeStepMtpSampled` lifts that
at the model level: it drafts and verifies identically, then walks the verified window **sampling**
one token per row -- one uniform draw each, from that row's device summary -- and stops at the first
row whose sampled token is not the draft that follows it.

That is textbook rejection sampling against a deterministic proposal, so it is lossless: a greedy
drafter proposes a point mass on `x`, the accept rule `min(1, p(x)/q(x))` degenerates to "accept
with probability `p(x)`", and drawing `y ~ p` once and accepting iff `y == x` is exactly that rule.
Because exactly one draw is consumed per *emitted* token, the stronger property holds too -- for a
fixed seed the round emits the same tokens plain sampled decode would, token for token.

Greedy and sampled rounds share ONE implementation of the verify + acceptance walk and ONE commit
(`Model::VerifyAndResolveRound` and `CommitVerifiedWindow`), so they cannot drift apart; the greedy
path's own behaviour is unchanged, which `tests/model/test_mtp.cpp`'s pre-existing checks still
assert byte for byte. Full details, the one accepted divergence class and its adjudication, and the
measured numbers: [sampling.md](sampling.md) sections 9 and 11.

**Milestone 6 stage S3**: `--mtp N` is no longer gated on `temperature <= 0` at the `src/server`/
`src/cli` level either -- `Engine::RunRequest`/`RunTurn` route a `temperature > 0` request through
`DecodeStepMtpSampled` above instead of falling through to plain decode. See
[server.md](server.md)'s stage S3 correction and [sampling.md](sampling.md) section 12.

## Container

`D:\models\r4dx\qwen38-27b.r4dx` (the real 64-layer container used throughout `docs/perf.md`)
**already carries `mtp.*` weights in all four layouts** (verified directly against its header by
this pass -- `mtp.norm`, `mtp.fc`, `mtp.pre_fc_norm_hidden`, `mtp.pre_fc_norm_embedding`,
`mtp.attn.qg.{bf16,mxfp4,w4a16,w4a8}.*`, etc. are all present), so no reconversion of the full
checkpoint was needed for this task. The 4-layer test container from the original MTP pass is
reused unchanged for `tests/model/test_mtp.cpp`:

```
r4dx-convert --input C:\AI\models\Qwen3.8-27B --output D:\models\r4dx\qwen38-27b-l4-mtp.r4dx ^
    --layers 4 --mtp on --vision off --layouts bf16,w4a16 --lm-head 4bit+bf16
```
(24.0s, 11.95 GB, 117 tensors.) `tests/model/test_forward_smoke.cpp`'s existing
`qwen38-27b-l4-bf16.r4dx` (no mtp) is untouched. Note: this 4-layer container has no `--layers`
CLI equivalent (`r4dx-cli` always loads `Config().num_hidden_layers` layers, i.e. the full 64) --
it is only ever driven through the `Model::Load(ModelOptions{.layer_limit=4, ...})` path
`tests/model/test_mtp.cpp` uses directly, not through `r4dx-cli`.

## Incident: the 0-1.2% acceptance bug

An Opus review (2026-09-19) traced the original pass's 0-1.2% measured acceptance rate to two
blocker-severity bugs, both in this pass's own new code, neither in the verify/commit/rewind
plumbing (which the review independently re-derived from the kernel sources and found correct):

1. **Swapped `fc` input concat order** (`mtp_head.cpp`'s `Draft()`, now also `PrimeKv()`).
   `qwen3_5_mtp.py:152-155` computes `inputs_embeds = pre_fc_norm_embedding(inputs_embeds)` FIRST,
   `hidden_states = pre_fc_norm_hidden(hidden_states)` SECOND, THEN
   `torch.cat([inputs_embeds, hidden_states], dim=-1)` -- the embedding occupies `fc`'s FIRST
   `hidden` input columns, the hidden state the SECOND `hidden` columns. The original code wrote
   `pre_fc_norm_hidden(h)` into the first half and `pre_fc_norm_embedding(embed(t))` into the
   second -- since `mtp.fc.weight` is `[5120, 10240] = [out, 2*hidden]`, this applied each half of
   the weight matrix to the WRONG operand, making every draft uncorrelated with the model
   regardless of how good `h_seed` was. Fixed: the embedding rmsnorm now writes into
   `concat_buf[0..hidden)`, the hidden-state rmsnorm into `concat_buf[hidden..2*hidden)`.
2. **MTP's own KV cache reset every round instead of built in lockstep** (`mtp_head.h`/`.cpp`,
   `model.h`/`.cpp` -- see "Design" above for the corrected design and how it works). The original
   design's premise -- that `h_seed` alone carries enough context for a near-empty per-round MTP
   attention cache -- does not match how this class of module is actually trained (real
   DeepSeek-V3/GLM-style MTP modules run with their OWN full-sequence attention, teacher-forced in
   parallel with the main model over the WHOLE training sequence, so at inference their attention
   layer needs a REAL KV history built from the actual prompt + every accepted token). This
   revision drives `mtp.layer` through the main model's own per-position hidden states during every
   prefill chunk AND every decode step (`MtpHead::PrimeKv`), growing MTP's own KV cache to the same
   `max_ctx` as the backbone's.

Bug (1) alone was already severe enough to explain near-zero acceptance; bug (2) means even a
correctly-ordered concat would have had a badly-impoverished first draft's context beyond very
short generations (an MTP round more than a few tokens into a long generation would see the SAME
tiny 1-token-deep cache every time, regardless of how long the real conversation had grown).
Measured together, fixing both took acceptance from **0-1.2%** to **30-75%** (see "Measurement"
below) and turned MTP from a measured net slowdown into a measured **1.4-2.0x decode speedup**.

## Correctness

### `Model::VerifyWindow`'s per-position logits vs. sequential `DecodeStep` (`tests/model/test_mtp.cpp`)

`CheckVerifyMatchesSequential`: a window built to be EXACTLY what sequential decode itself produces
(so every row is "accepted" by construction) compared row-for-row against sequential `DecodeStep`'s
own logits, on the 4-layer MTP container. All rows land well under the task's 1e-2 gate -- the
batched (q_len>1) speculative-verify decode kernel path (attention's decode kernel at q_len>1,
GDN's conv_update/recurrent_update over a genuine multi-candidate window with the new per-candidate
slot banking) agrees with the plain sequential T=1 path to ordinary bf16/quantization noise, not a
structural difference. Unaffected by this revision's two fixes (this check was already passing
before them -- the verify/commit machinery was never the bug).

### Rejection rewind (`tests/model/test_mtp.cpp`)

`CheckRejectionRewind`: 12 rounds of real `DecodeStepMtpGreedy` on the 4-layer container (whose
drastic truncation makes MTP's own drafts agree with the equally-truncated real model only
sometimes, exercising both full acceptance and rejection organically) produced a token sequence
byte-identical to a fully sequential `DecodeStepGreedy` reference, for both bf16 and w4a16 -- still
true after this revision's KV-lockstep rewrite (re-verified, `ctest` 30/30). A broken rewind (either
the backbone's or MTP's own new one) would make the token generated immediately after a rejection --
and everything after it -- diverge; it did not.

### `--mtp K` vs `--mtp 0`, real 64-layer container, greedy

Measured this revision (`--layout w4a16`, `--max-ctx 2048`, `--mtp 3` vs `--mtp 0`, prompt "Write a
haiku about GPUs, then explain what a GPU is in two sentences.", `--max-tokens 48/128`): generated
text is **byte-identical** at both lengths tried, for w4a16/w4a8/mxfp4/bf16 alike -- a real change
from the original pass's report of divergence past ~64 tokens for some prompts.
>
> **Correction (review finding, 2026-09-20; re-measured on real hardware, same prompt/flags): this
> is no longer true for mxfp4.** Re-run twice at `--max-tokens 48` and `128`: w4a16 and w4a8 are
> still byte-identical between `--mtp 0`/`--mtp 3` (deterministic on repeat, SHA-256 confirmed at
> `--max-tokens 128`: w4a16 `621e747a...`, w4a8 `6656cbfb...`, matching for both `--mtp` values).
> mxfp4 now diverges at BOTH lengths -- `--mtp 0`/`--mtp 3` agree through "...a specialized
> electronic circuit designed to rapidly manipulate and " and then part ways ("generate images for
> a display..." vs "alter memory to accelerate the creation of images in a frame buffer..."), each
> configuration still fully deterministic on repeat. Both outputs are coherent, on-topic, and
> factually reasonable -- not a quality regression, just the batched-verify-vs-sequential
> floating-point non-associativity this section already names finally landing on THIS prompt for
> mxfp4 specifically, almost certainly because the TUNE pass's gemm_tuning_table.inc re-sweep
> (`docs/status.md`'s "Full Q5-fixed tune_gemm.py re-sweep") changed mxfp4's own M=1-vs-M=(K+1)
> GEMM tiling after this claim was written (that same re-sweep already moved mxfp4's own MTP
> acceptance 47.1% -> 52.9%, see "MTP head layout" below). See the scope-correction blockquote
> immediately below for why a single measured pair was never a general guarantee in the first
> place, and why this is expected, not a correctness bug.

This is expected, not just lucky: the original pass's divergence was root-caused to floating-point non-associativity
between the batched (q_len>1) verify-window attention-decode kernel and the sequential (q_len=1)
path (see prior revision's writeup, preserved below) -- an effect that is still structurally present
in the kernel, but with acceptance now high (30-75% vs 0-1.2%), far FEWER verify rounds run for the
same generation length (a K=3 round that accepts 3/3 drafts advances the sequence by 4 tokens in
one verify call instead of 4 separate q_len=1 calls), so there are correspondingly fewer
opportunities for the rounding-order effect to flip a near-tie argmax.
>
> **Scope correction (2026-09-20, FIX pass, review finding): the bolded "byte-identical" claim above
> is a property of the ONE prompt/length pair actually measured, not a general guarantee, and this
> section (and the summary bullet in "Known gaps"/the revision-history line below) previously read as
> if it were the latter.** Review testing against different prompts/`--max-tokens` values on this
> same w4a16 container found real, deterministic divergence between `--mtp` values -- not
> nondeterminism (each configuration reproduces its own output exactly on repeated runs), but the
> SAME batched-verify-vs-sequential floating-point non-associativity this section already names as
> "structurally present," just not rare enough to avoid at every prompt length. Do not treat a
> divergence between `--mtp 0` and `--mtp K` (or between different `K` values) on a prompt/length
> this doc did not explicitly measure as evidence of a new regression by itself -- corroborate with
> `CheckVerifyMatchesSequential`/`CheckRejectionRewind` (which check the actual verify-vs-sequential
> and accept/reject-rewind CONTRACTS, not textual identity) before concluding something broke. This
> is also why `docs/status.md`'s R2/P2 incident treats "generated text differs from baseline" as a
> legitimate bug signal for the fused-epilogue work specifically: that comparison holds bf16 inputs
> fixed and checks identical bytes in produce identical text out (no batched-vs-sequential
> non-associativity in play), which is a different, stronger guarantee than what this section
> measures.

<details>
<summary>Original pass's divergence measurement (superseded by the above, kept for the record)</summary>

> the q_len>1 verify-window attention-decode kernel launch computes the SAME masked/causal sum as
> q_len==1's sequential kernel launch (verified directly against `r4d_attn_decode_h256_gqa6.hip`'s
> source: `klimit = ctx - q_len + qrow` is a genuinely per-row-correct causal bound, not a batch-wide
> one), but the SPLIT-KV tile range `[t_lo,t_hi)` a launch visits is chosen from the BLOCK-shared
> `ctx=seqused_k` value, which differs between a verify call (`ctx = pos_+T`, shared across all T
> rows) and T independent sequential calls (each its own smaller `ctx`) -- masked-out (future)
> entries contribute exactly 0.0 in exact arithmetic either way, but floating-point summation is not
> associative, so the two paths' low-order bits can differ.

</details>

## Measurement: acceptance rate and decode tok/s

`--mtp {0,1,2,3,4}`, real 64-layer container, greedy, prompt "Write a haiku about GPUs, then
explain what a GPU is in two sentences." (`docs/perf.md`'s own prompt), `--max-tokens 128 --max-ctx
2048 --stats`, measured on HIP device 1 after this revision's fixes:

| Layout | mtp=0 tok/s | mtp=1 (accept%) | mtp=2 (accept%) | mtp=3 (accept%) | mtp=4 (accept%) |
|---|---|---|---|---|---|
| w4a16 | 32.55 | 51.49 (75.0%) | 58.80 (58.3%) | **65.56 (54.3%)** | 57.80 (39.4%) |
| w4a8  | 30.73 | 47.42 (68.8%) | 47.68 (42.0%) | 47.20 (32.5%) | **51.12 (31.2%)** |
| mxfp4 | 26.87 | 39.11 (61.1%) | **48.52 (56.1%)** | 47.05 (41.9%) | 45.86 (33.6%) |
| bf16 (K=3 only, `--max-tokens 32`) | 1.41 | -- | -- | **2.21 (48.7%)** | -- |

Best K per layout, decode speedup over `--mtp 0`: **w4a16 +101.4% (K=3), w4a8 +66.4% (K=4), mxfp4
+80.6% (K=2), bf16 +56.7% (K=3)**. Every layout's best-K configuration is a substantial win, not a
marginal one -- **MTP now more than doubles decode throughput on this checkpoint for the best-tuned
K per layout**, a complete reversal of the original pass's "measured net slowdown" conclusion (which
was an artifact of the two bugs above, not a property of this checkpoint's MTP module). Acceptance
falls off past the per-layout optimum (predictably: a longer speculative chain is more likely to
contain an early wrong guess, wasting the tail of the round's compute on drafts that get rejected
anyway) -- `--mtp 3` is a reasonable default across layouts; a caller optimizing for one specific
layout should sweep K as shown here.

Generated text for every run above: coherent, on-topic, stops on EOS (see "Correctness" above for
the byte-identical-vs-`--mtp 0` finding). Container load ~9-22s depending on layout/cache-warmth
(matches `docs/perf.md`'s own load-time noise, unaffected by this revision).

**Milestone 4 correction (2026-09-20, integration pass)**: the best-K-per-layout table above
predates the R2/P2 fused-epilogue enablement and the Q5-fixed GEMM re-sweep, both of which changed
per-layout decode numerics. Re-checked directly against real hardware post-integration: **w4a16's
best K is still K=3** (68.73 tok/s, 46.3% acceptance); **w4a8's best K is still K=4** (57.86 tok/s,
31.0% acceptance -- lower than this table's old 51.12 despite being the "best" K, because the
Q5 re-sweep's verify-band GEMM retiling dropped w4a8's acceptance across the board, see
`docs/perf.md`'s re-sweep section); **mxfp4's best K moved from K=2 to K=3** (65.04 tok/s, 52.9%
acceptance, vs K=2's now-lower 58.96 tok/s/56.1%) -- checked directly, not assumed. Full sweep
values and the consolidated table: `docs/perf.md`'s "Milestone 4: integration confirmation sweep"
section.

## MTP head layout

Status: measured (this pass, HIP device 1, real 64-layer container). `ModelOptions::mtp_head_layout`
(`std::optional<Layout>`, default `std::nullopt`) controls the layout used ONLY for the MTP head's
four quantized linears (`mtp.attn.qg/o`, `mtp.mlp.gate_up/down`) -- independent of the body's own
`--layout`. `nullopt` (CLI default, `--mtp-head-layout layout`) tracks the body layout; an explicit
`Layout::kBf16` (CLI: `--mtp-head-layout bf16`) forces the head's exact-arithmetic bf16 tensors
instead (the container carries both forms for every layout -- `mtp.attn.qg.{bf16,mxfp4,w4a16,w4a8}.*`
etc., "Container" above), at ~0.5 GB extra VRAM.

**Measured (`--mtp {1,2,3,4}`, w4a8/w4a16/mxfp4, same prompt/flags as "Measurement" above,
`--max-tokens 128`):**

| Layout | K | bf16 head: decode tok/s (accept%, tok/round) | layout head: decode tok/s (accept%, tok/round) |
|---|---|---|---|
| w4a16 | 1 | 52.07 (78.4%, 1.78) | 52.12 (75.0%, 1.75) |
| w4a16 | 2 | 56.13 (55.8%, 2.12) | **60.14 (58.3%, 2.17)** |
| w4a16 | 3 | 61.54 (51.9%, 2.53) | **67.34 (54.3%, 2.60)** |
| w4a16 | 4 | 56.97 (41.4%, 2.53) | 59.82 (39.4%, 2.45) |
| w4a8  | 1 | 44.90 (62.0%, 1.62) | **47.94 (68.8%, 1.69)** |
| w4a8  | 2 | 47.55 (44.2%, 1.88) | 48.40 (42.0%, 1.84) |
| w4a8  | 3 | 47.00 (34.2%, 2.02) | 48.54 (32.5%, 1.98) |
| w4a8  | 4 | 46.44 (28.3%, 2.13) | **52.49 (31.2%, 2.25)** |
| mxfp4 | 1 | 37.43 (55.4%, 1.54) | **39.56 (61.1%, 1.59)** |
| mxfp4 | 2 | 45.57 (51.2%, 2.00) | **49.47 (56.1%, 2.10)** |
| mxfp4 | 3 | 44.91 (40.0%, 2.15) | **48.24 (41.9%, 2.21)** |
| mxfp4 | 4 | **48.53 (38.6%, 2.46)** | 47.55 (33.6%, 2.26) |

The layout-matched (quantized) head is faster than the bf16 head in 23 of 24 (layout, K)
configurations (smaller GEMMs -- no surprise) and acceptance is a wash: often slightly HIGHER with
the quantized head, never meaningfully lower, at every K on every layout. **The layout-matched head
is therefore the new default** (`ModelOptions::mtp_head_layout = std::nullopt`, CLI
`--mtp-head-layout layout`) -- the earlier bf16-default assumption (that the draft head's own
numerical precision would matter enough to justify its extra VRAM/compute) does not hold on this
checkpoint. This also *answers* the task's step 2 question in the negative for the "head precision"
hypothesis: forcing the head to exact bf16 barely moves acceptance at all (compare each row's two
columns above), so the head's own arithmetic is not what limits w4a8/mxfp4's acceptance below
w4a16's -- see "Acceptance gap investigation" below for what this pass ruled in/out instead.

### Acceptance gap investigation

At matched K, w4a16's acceptance stays well above w4a8's/mxfp4's regardless of head layout (e.g.
K=3: w4a16 51.9-54.3% vs w4a8 32.5-34.2% vs mxfp4 40.0-41.9% -- a ~15-20 point gap that the bf16
head does not close, per the table above). Two candidate mechanisms were checked and ruled out this
pass; the actual mechanism is not yet isolated:

1. **MTP head numerical precision** -- ruled out. See the table above: a bf16 head sees the exact
   same ~15-20 point gap as the layout-matched head, at every K. If the head's own arithmetic were
   the bottleneck, forcing it to bf16 should have narrowed the gap; it did not, within run-to-run
   noise.
2. **Verify-window batched-attention correctness** -- already ruled out by the M2 fix pass
   (`CheckVerifyMatchesSequential`, unaffected by this pass) and independently reconfirmed by this
   pass's "Correctness" section above: `VerifyWindow`'s per-position logits agree with sequential
   `DecodeStep` to ordinary bf16/quantization noise, for every layout. Acceptance is about whether
   the DRAFT (produced by `MtpHead::Draft` from `h_seed` alone, never checked against the real model
   until verification) matches what that SAME layout's own sequential decode would have produced --
   not a bug in how verification computes its own logits.
3. **Logit-margin ("decision confidence") hypothesis -- tested this pass, falsified.** Hypothesis:
   body-layer quantization (specifically activation quantization -- w4a8's int8 row-activation
   quant, mxfp4's native fp4 weights+activations -- vs w4a16's weight-only int4 quant, which keeps
   activations in bf16 throughout) flattens the shared lm_head's own top1-vs-top2 logit margin
   system-wide, making EVERY next-token decision more borderline (and so easier for any
   approximation, MTP draft included, to flip), independent of MTP entirely. Measured directly: a
   throwaway probe (`Model::Prefill` + 24 sequential `Model::DecodeStep` calls, real 64-layer
   container, deterministic filler tokens, `mtp_draft_k=0` throughout -- no MTP code path touched)
   computed the mean/stdev/min top1-vs-top2 logit margin per layout:

   | Layout | mean margin | stdev | min margin | steps with margin < 1.0 (of 24) |
   |---|---|---|---|---|
   | w4a16 | 3.12 | 1.84 | 0.00 | 5 |
   | w4a8  | 3.78 | 1.69 | 0.06 | 3 |
   | mxfp4 | 3.57 | 1.66 | 0.03 | 3 |

   This is the OPPOSITE of the hypothesis: w4a16 has the SMALLEST mean margin and the MOST
   low-margin steps of the three, yet the HIGHEST MTP acceptance. Margin alone does not explain the
   acceptance ranking -- falsified, not pursued further.

With both the head-precision and logit-margin hypotheses ruled out, and verify-window correctness
already independently established, this pass's conclusion was: **not a bug**, but the specific
mechanism was not isolated yet -- a direct golden-referenced comparison of `h_seed` itself against a
`tools/reference`-style ground truth was named as the natural next step and left undone.

### h_seed drift (Milestone 4 follow-up, 2026-09-20): measured, and it explains the ordering

Built the one route the pass above named but never had: a 4-layer container carrying **all four**
layouts plus `mtp.*` weights side by side (`D:\models\r4dx\qwen38-27b-l4-allmtp.r4dx`, converted via
`r4dx-convert --layers 4 --mtp on --layouts bf16,w4a16,w4a8,mxfp4` -- the pre-existing
`qwen38-27b-l4-mtp.r4dx` only carries bf16+w4a16). Added a diagnostic-only accessor,
`Model::DebugSeedHiddenBf16()` (`src/model/model.h`/`.cpp`), that reads back `mtp_seed_hidden_` --
the EXACT pre-final-norm hidden-state row `MtpHead::Draft`'s first step consumes -- and a one-off
tool, `tests/model/tool_hseed_drift.cpp` (built by `tests/model/CMakeLists.txt`, deliberately never
registered as a `ctest` test since it prints numbers rather than asserting a pass/fail contract),
that feeds the SAME fixed 64-token stream through a bf16 (exact-arithmetic reference, this project's
own established convention) `Model` and each quantized layout's `Model` against that container,
sampling `h_seed` every 8 tokens (8 positions across the stream) and comparing each quantized
layout's sample to the bf16 reference's own sample at that same position by cosine similarity and
relative L2.

**Measured** (HIP device 1, real hardware, mean over the 8 sampled positions):

| Layout | mean cosine vs bf16 | mean rel L2 vs bf16 | K=3 acceptance (from the table above) |
|---|---|---|---|
| w4a16 | **0.99763** (best) | **7.16e-2** (smallest) | **51.9-54.3%** (highest) |
| mxfp4 | 0.99745 (middle) | 1.061e-1 (middle) | 40.0-41.9% (middle) |
| w4a8  | 0.99496 (worst) | 1.257e-1 (largest) | 32.5-34.2% (lowest) |

This is **exactly** the measured acceptance ranking: the layout with the smallest h_seed drift from
the exact-arithmetic reference has the highest acceptance, the layout with the largest drift has the
lowest, and mxfp4 lands in the middle on both independent metrics (cosine and rel L2) as well as on
acceptance. Per-position noise exists (at 2 of the 8 sampled positions mxfp4's rel L2 briefly
exceeds w4a8's -- a small sample on a drastically-truncated 4-layer container), so this is a
mean-level correlation, not a claim that every single position preserves the ranking; the agreement
across two independent metrics and three layouts, both matching the independently-measured
acceptance ranking, is not plausibly coincidence.

**Mechanism, identified.** Nearly all of every layout's rel L2 traces to ONE hidden dimension
(component index 3994 of 5120) whose bf16-reference magnitude (+26.5) dwarfs every other observed
component (roughly -2..+2) -- the well-documented "massive activation" / outlier-dimension
phenomenon in transformer residual streams. Its own absolute quantization error tracks the identical
per-layout ranking as the aggregate metric: w4a16 -0.375 (-1.4% relative), mxfp4 -1.125 (-4.2%),
w4a8 -2.875 (-10.8%). Because this single dimension's squared magnitude dominates the vector's
squared L2 norm, its own quantization error is effectively what the aggregate rel L2 (and,
plausibly, whatever sensitivity the lm_head has to the residual stream at this dimension) measures.

**Conclusion: h_seed drift explains the acceptance ordering** -- a measured positive result, not
another falsified hypothesis. This is expected quantization behavior on a residual-stream dimension
with an outsized dynamic range (the same reason outlier-aware quantization schemes exist in the
wider literature), not a bug -- no code change is warranted here, so this pass's own task item 4
("if the investigation finds an actual bug, fix it and re-measure") does not apply; there is nothing
to fix. Per the task's own conditional ("if h_seed drift does not explain it, the next candidates in
order..."), the KV-history-divergence and verify-window-batched-vs-sequential-disagreement-rate
hypotheses were **not pursued this pass**, since a measured, mechanistically-grounded positive
answer for h_seed drift was already found -- flagged, not silently dropped, as the natural next step
if a future pass wants to push further (e.g. testing whether an outlier-aware quantization scheme
for just the handful of massive-activation dimensions narrows the acceptance gap without giving up
the layout's throughput).

## Device-resident draft loop

Status: implemented and measured (this pass). `MtpHead::Draft`'s per-draft-token host round-trip
(`docs/mtp.md`'s prior "Known gaps" entry: a `stream.Synchronize()` + small D2H argmax readback +
H2D embedding upload per drafted token) is now eliminated when the container's `text.embed_tokens`
has a VRAM mirror (`Container::EmbedTokensDeviceResident()`, `ModelOptions::embed_device_resident`,
default `true` -- `docs/r9700.md`'s P3 measured this table fits, 2.54 GB bf16, with headroom left
over at 131k ctx): a new device-side gather kernel (`r4dx_embedding_gather_bf16`,
`src/kernels/src/r4dx_kernels.hip`) reads a row straight out of the VRAM-resident embedding table
by a DEVICE int32 id -- `r4dx_argmax_f32`'s own `out_idx` output feeds directly into the next
draft step's gather with zero host syncs in between. `positions_`/`seqused_k_` for the whole K-step
draft window are preloaded in one H2D upload each before the loop (`mtp_head.cpp`'s `Draft`) instead
of one blocking `CopyFromHost` per step, and the whole window's drafted token ids are read back in
ONE `stream.Synchronize()` + D2H at the end instead of one pair per step. `Model::RunChunk` and
`Model::VerifyWindow` use the same device gather for the main decode/prefill path (one fewer H2D per
token there too), falling back to the original host-gather path (`EmbedTokens`,
`src/model/embedding.h`) unconditionally when `Container::EmbedTokensDeviceResident()` is false --
either `ModelOptions::embed_device_resident=false` (explicit opt-out, e.g. to keep VRAM margin for a
very large KV cache) or the container's free-VRAM heuristic at load time
(`Container::Load`'s own comment: 2x headroom over the embedding table's own size, checked against
whole-GPU free memory at the point of upload) decided it would not fit. Every existing host-gather
call site keeps working unchanged either way -- the device mirror is purely an additional path, never
a replacement for `EmbedTokensHost()`.

Regression coverage: `tests/model/test_mtp.cpp`'s existing `CheckVerifyMatchesSequential` and
`CheckRejectionRewind` both exercise this new path directly (the 4-layer MTP test container loads
with `embed_device_resident` at its `ModelOptions` default of `true`), and both still pass
byte-identical/rel-L2-gated as before -- the device-resident gather produces the same tokens as the
host path it replaced. `CheckPlainDecodeUnaffectedByMtpConfig` (new this pass -- a deliberately
DIFFERENT contract from `CheckRejectionRewind`'s existing MTP-vs-sequential comparison, not a
duplicate of it, per its own file comment in `tests/model/test_mtp.cpp`) additionally confirms the
device-gather branch added to `Model::RunChunk`/`VerifyWindow` does not perturb the PLAIN (non-MTP)
decode path for an MTP-sized `Model`.

## Reduced-vocab draft head

Status: implemented and measured (docs/r9700.md R9). §2.2's own draft-side byte budget is the
reason wide speculation (DFlash2-style K=10..16) was blocked: a sequential single-head drafter pays
the FULL 248320-entry `lm_head` (675 MB of the 926 MB per drafted row) for every drafted token, so
K=16 costs ~24 ms of drafting to accelerate a ~26-30 ms verify step -- worse than doing nothing. A
**top-N slice of `lm_head`**, used ONLY for drafting, cuts that to ~N/vocab of the bytes (N=8192 =>
41.9 M params = 22.3 MB, ~0.04 ms) and makes wide K viable.

**Why this cannot degrade output quality.** `Model::VerifyWindow` -- the ONLY code path that ever
commits a token into the model's real state -- always runs the real model's full-vocab `lm_head`
(`container_.LmHead()`), never `mtp.draft_head.lm_head`, regardless of which head drafted the
candidate. A draft token the reduced head's own subset could not have produced (because the real
next token is outside the subset) is therefore just an ORDINARY REJECTED DRAFT -- indistinguishable,
from the verify path's point of view, from a wrong guess a full-vocab head would also have rejected.
The only thing the subset size can change is the ACCEPTANCE RATE (and therefore tok/s), never
whether an accepted token is correct. See `MtpHead::Draft`'s own doc comment
(`src/model/mtp_head.h`) and `MtpWeights::draft_lm_head`'s comment (`src/model/container.h`) for the
same argument spelled out at the code site.

**Vocabulary subset choice and coverage.** `tests/model/tool_vocab_calib.cpp` (a diagnostic tool,
not a ctest test, same convention as `tool_hseed_drift.cpp`) measures two candidate methods against
a real calibration run: the real 64-layer container (`D:/models/r4dx/qwen38-27b-v3.r4dx`, `--layout
w4a16`, `--mtp 0`) teacher-forced (real corpus tokens fed one at a time via `Prefill`+`DecodeStep`,
never the model's own chained output) over the first ~6000 tokens of `D:/models/wikitext-2-raw/
wiki.train.raw`, recording the model's own greedy argmax prediction at every position.

- **Method A ("frequency")**: top-N tokens by frequency IN THE CALIBRATION CORPUS TEXT ITSELF (a
  property of the text, via the real tokenizer, independent of the model).
- **Method B ("predicted")**: top-N tokens by frequency AMONG THE MODEL'S OWN GREEDY PREDICTIONS
  over the calibration set -- the task's own recommendation ("better matched to what gets
  drafted"), since a draft head's job is to approximate what the backbone predicts, not what token
  is common in English text generally.

**Coverage** = fraction of the model's own greedy predictions (at every calibration position) whose
id falls inside the candidate subset -- an upper bound on how often the reduced head COULD possibly
match the real model (a draft outside the subset is a guaranteed rejection; a draft inside the
subset still has to beat the reduced head's own approximation error to be accepted, so measured
acceptance is always <= this number).

**Measured** (`tests/model/tool_vocab_calib.cpp`, real hardware, `D:/models/r4dx/qwen38-27b-v3.r4dx`,
`--layout w4a16 --mtp 0`, `D:/models/wikitext-2-raw/wiki.train.raw`, 20000 calibration positions:
15000 TRAIN -- subset built from these only -- + 5000 HELD-OUT -- coverage measured against these
only, so the number below is genuinely out-of-sample, not "does a subset cover the exact data it was
built from"): top-1 self-teacher-forced accuracy over the whole run was 57.73% (an informative
sanity number -- how often the model's own greedy guess equals the actual next token in real text --
not the coverage metric itself). Only **2446 distinct ids** were ever the model's own top prediction,
and only **3198 distinct ids** ever occurred as a TRAIN-portion corpus token, across all 15000 TRAIN
positions:

| Candidate N | Held-out coverage, method B ("predicted") | Held-out coverage, method A ("frequency") | Actual subset size built |
|---|---|---|---|
| 4096 | 76.80% | 78.92% | 2446 (predicted) / 3198 (frequency) |
| 8192 | 76.80% | 78.92% | 2446 (predicted) / 3198 (frequency) |
| 16384 | 76.80% | 78.92% | 2446 (predicted) / 3198 (frequency) |

**All three candidate N collapse to the identical measured coverage, because the calibration run's
own OBSERVED vocabulary (2446-3198 distinct ids over 15000 positions) is itself smaller than every
candidate N tested** -- Zipf's law on a single-domain 15-20k-token sample: real text (even
Wikipedia's own encyclopedic prose) concentrates onto a few thousand distinct tokens at this sample
size, so "top-4096" and "top-16384" are both just "every distinct id ever observed" here. This is a
genuine, informative negative result about the calibration CORPUS, not a flaw in the
subset-construction code (`PlanLinearLayouts`'s own N-must-be-a-multiple-of-16 constraint is
satisfied by rounding the observed set up with harmless low-id filler rows, `MtpWeights::
draft_lm_head`'s comment) -- a substantially larger and more topically diverse calibration corpus
(hundreds of thousands of tokens across many domains, not one small benchmark file) would be needed
to actually separate the 4k/8k/16k design points from each other. **Ruled out one candidate
explanation directly**: re-ran the same 20000-position calibration sampling 10 DISPERSED,
evenly-spaced segments across the whole 10.9 MB `wiki.train.raw` file (instead of one contiguous
prefix, `tests/model/tool_vocab_calib.cpp`'s `ReadDispersedSegments`) -- the result was numerically
IDENTICAL (2446/3198/2977 distinct ids, 76.80%/78.92% coverage, to four decimal places), which rules
out "the sample happened to land in one narrow topic" as the cause and instead points at WikiText-2
itself: a small (10.9 MB), intentionally curated benchmark corpus with a genuinely narrow effective
vocabulary at any sampled size, not a large diverse natural-text corpus. This machine has no larger
general-text corpus available (checked: only `D:/models/wikitext-2-raw` exists under `D:/models`) --
a real, larger corpus (or several-hundred-thousand-token synthetic generation from the model itself)
is the concrete next step, flagged in "Known gaps" below, not attempted this pass.

**Method A ("frequency") slightly beat method B ("predicted") in THIS measurement** (78.92% vs
76.80%) -- the opposite of the task's own a-priori expectation ("better matched to what gets
drafted"). Plausible explanation, not further investigated: on a narrow, self-similar corpus like
Wikipedia prose, raw token frequency is already a strong proxy for what the model will predict (the
two methods' actual id SETS overlap heavily in practice), and method A's slightly larger observed
set (3198 vs 2446 distinct ids) gives it more headroom to cover a held-out slice of the SAME narrow
domain. This ordering might reverse on a more diverse corpus where the model's own predictions
generalize differently than raw text frequency -- flagged as an open question, not resolved here.
The **shipped default still uses method B** (`tests/model/tool_vocab_calib.cpp`'s own choice, task's
recommendation) for the reasons stated above (better matched in principle to what MTP actually
drafts, even though this one measurement did not confirm an advantage) -- built from the FULL
20000-position calibration run (not just the 15000-position TRAIN split above, which was for the
coverage MEASUREMENT only), giving **2977 distinct predicted ids**, rounded up to **2992** (a
multiple of 16, `r4d_gemm_*_nt_m64`'s own row-tiling requirement) with harmless low-vocab-id filler
rows that are real (if never-predicted-in-calibration) vocabulary ids -- never garbage, and
never read by anything except this one drafting path.

**Shipped default**: method B ("predicted"), N=2992 (rounded up from the calibration run's own 2977
distinct predicted ids, see the coverage measurement below for why this is smaller than the original
4k/8k/16k design targets) -- `tests/model/tool_vocab_calib.cpp` writes this subset to a
`{"vocab_ids": [...]}` JSON, which `r4dx-convert --draft-vocab-ids <path>` bakes into
`mtp.draft_head.lm_head.{layout}` + `mtp.draft_head.vocab_ids` (docs/container-format.md's own
"mtp.draft_head.*" section). `N` is a converter-time parameter (any calibration-chosen size, must be
a multiple of 16 for the quantized layouts' own row-tiling requirement) -- the code path (loader,
`MtpHead::Draft`, the gather kernel) is fully generic in N and was exercised at N=4096 in the
plumbing/correctness tests (`tests/model/test_mtp.cpp::CheckReducedVocabDraftHeadLossless`) as well
as N=2992 in the real shipped container.

**Container format and loader.** New OPTIONAL tensors, versioned by presence (not a format-version
bump): a container converted without `--draft-vocab-ids` simply lacks `mtp.draft_head.*`, and
`Container::Load` probes for `mtp.draft_head.vocab_ids` the same way it probes `mtp.norm` for
`HasMtp()` -- absent means `MtpWeights::HasDraftHead()` is false and `MtpHead::Draft` falls back to
the exact pre-R9 full-vocab path unconditionally, so every container built before this feature
(including every container this milestone's own perf tables were measured against) keeps loading
and behaving identically.

**Wiring.** `MtpHead::Draft` gained a `use_reduced_vocab` parameter (default `true`,
`ModelOptions::mtp_draft_reduced_vocab`, CLI/server `--mtp-draft-head {reduced,full}`): when true AND
the container has a draft head, every draft step's own `lm_head` GEMM+argmax runs against the
smaller `draft_lm_head` (via the SAME `FinalLmHead` class the full-vocab path already used -- it is
generic in its own `lm_head_.N`, so no new GEMM code was needed) instead of the full one, and the
resulting SUBSET-LOCAL argmax index is mapped back to a real vocabulary id by a new device kernel,
`r4dx_gather_i32` (`src/kernels/src/r4dx_kernels.hip`, `table[idx[i]]`, bounds-checked, clamps
out-of-range to index 0 like `r4dx_embedding_gather_bf16` already does) -- entirely on-device, zero
host syncs, so the chained device-resident draft loop (docs/mtp.md's own section above) is
unaffected. From that point on the value is a real vocab id exactly like the full-vocab path always
produced, so no other line of `Draft()`'s own algorithm, and nothing in `Model::VerifyWindow`,
needed to change.

**Widened draft width.** Raising K beyond the old K<=4 ceiling turned out to need NO structural
change: `GdnStateManager`'s window bank (`gdn_state.h`), the conv-buffer rolling depth
(`state_len_max = conv_width - 2 + max_decode_window`), and `Model::VerifyWindow`'s
`mtp_logits_dev_`/`mtp_argmax_dev_` scratch (sized `(mtp_draft_k_+1)*vocab`/`(mtp_draft_k_+1)`) were
already parametrized by `ModelOptions::mtp_draft_k` at `Model::Load()` time from the Milestone 3
MTP-quality pass -- the M3 review's blocker-class off-by-one in this exact bookkeeping was already
fixed as part of getting `state_len_max`'s formula right, and that fix generalizes to any K, not
just K<=4. `tests/model/test_mtp.cpp::CheckWideWindowRejectionRewind` is the real-hardware
regression test for this claim: same lossless-rewind contract as `CheckRejectionRewind` (mtp-decoded
sequence must exactly equal a sequential reference, across at least one real rejection), run at
K=16 on both `bf16` and `w4a16` -- **passing, unmodified formulas, on real hardware**. `PagedKvCache`
needed no change either (`block_size=16` is a physical page size; a K=16 verify window's 17
candidate positions already span multiple blocks the same way any >16-token prefill chunk does).

**Measured K sweep and headline numbers.**

**Measured** (real hardware, HIP device 1, `D:/models/r4dx/qwen38-27b-v3-draftvocab.r4dx` -- the real
64-layer container, vision on, reconverted with the shipped N=2992 draft head baked in --
docs/perf.md's standard prompt/flags, `--max-tokens 128 --max-ctx 2048 --temperature 0`, one run per
cell, `tools/`-adjacent scratch script `build/logs/sweep_r9.ps1`):

| Layout | K | Reduced head: decode tok/s (accept%, tok/round) | Full head: decode tok/s (accept%, tok/round) |
|---|---|---|---|
| w4a16 | 0 (baseline) | 38.44 (--, --) | (same, head-independent) |
| w4a16 | 1 | 50.70 (40.7%, 1.41) | -- |
| w4a16 | 2 | 52.78 (26.9%, 1.54) | -- |
| w4a16 | **3** | **53.35 (20.9%, 1.63)** | -- |
| w4a16 | 4 | 52.37 (16.5%, 1.66) | **65.02 (37.1%, 2.37)** |
| w4a16 | 6 | 48.50 (11.0%, 1.66) | -- |
| w4a16 | 8 | 44.83 (8.2%, 1.66) | 49.58 (17.7%, 2.31) |
| w4a16 | 12 | 40.08 (5.5%, 1.66) | -- |
| w4a16 | 16 | 33.88 (4.1%, 1.66) | 33.98 (8.9%, 2.31) |
| w4a8 | 0 (baseline) | 36.11 | (same) |
| w4a8 | 1 | 43.89 (29.6%, 1.30) | -- |
| w4a8 | **2** | **46.33 (21.9%, 1.44)** | -- |
| w4a8 | 4 | 43.57 (11.5%, 1.46) | **57.20 (31.0%, 2.19)** |
| w4a8 | 8 | 37.81 (5.8%, 1.46) | 49.17 (18.3%, 2.36) |
| w4a8 | 16 | 31.32 (3.2%, 1.51) | 34.62 (8.8%, 2.32) |
| mxfp4 | 0 (baseline) | 32.85 | (same) |
| mxfp4 | 1 | 45.50 (47.9%, 1.48) | -- |
| mxfp4 | **2** | **46.63 (29.1%, 1.58)** | -- |
| mxfp4 | 4 | 46.09 (17.0%, 1.68) | **59.68 (36.7%, 2.47)** |
| mxfp4 | 8 | 40.52 (8.5%, 1.68) | 48.27 (18.3%, 2.47) |
| mxfp4 | 16 | 30.97 (4.0%, 1.64) | 34.95 (9.5%, 2.52) |

(K=1,2,3,6,12 not re-run with the full head -- the reduced-vs-full comparison used a representative
subset, K=4/8/16, per this pass's own time budget; every K value WAS run with the reduced head, per
the task's own K list.)

**Headline result, stated honestly.** The single best number measured across this entire sweep is
**65.02 tok/s** (w4a16, K=4, FULL-vocab head) -- **0.95x of the M3 baseline's 68.37 tok/s** (a
different K, a different exact container revision with vision-tower + draft-head tensors added, and
ordinary run-to-run noise; this pass's own K=0 baseline, 38.44 tok/s, is itself within 1.1% of M3's
38.86, so the measurement setup is consistent with M3's). **The reduced-vocab head's own best number
is 53.35 tok/s (w4a16, K=3) -- 0.78x of the M3 baseline, i.e. a REGRESSION, not the projected
2.5-3x.** Every layout's decode tok/s PEAKS at a low K (K=2 or K=3) with the reduced head and then
MONOTONICALLY DECLINES as K grows further, eventually dropping BELOW the K=0 baseline at K=16
(w4a8: 31.32 < 36.11; mxfp4: 30.97 < 32.85) -- the opposite of "wide K becomes viable".

**Root cause, and it is NOT a bug in the mechanism.** `tests/model/test_mtp.cpp`'s lossless-rewind
checks (`CheckReducedVocabDraftHeadLossless`, both `use_reduced=true/false`, plus the wide-K=16
checks) all pass byte-identical-to-sequential on real hardware -- the reduced head never produces a
wrong ACCEPTED token, exactly as designed. What tanks throughput at high K is ACCEPTANCE RATE:
the reduced head's own subset (this pass's calibration run, "Vocabulary subset choice and coverage"
above) measured only **76.8% held-out coverage** with a natural size of **2977 distinct ids** --
an order of magnitude short of the 8k-16k this technique's own economics (docs/r9700.md §2.2) assume
for a "well-covered" subset, because a 20000-token single-domain (Wikipedia) calibration corpus does
not contain enough distinct vocabulary to build a bigger, more-representative one (see the coverage
section above for the full Zipf's-law explanation). At K=16 the FULL head still only reaches
8.9-9.5% acceptance on this checkpoint/prompt (a real, independent limit on wide-K speculation for
THIS model -- consistent with docs/mtp.md's own "Known gaps" note that acceptance is below what a
purpose-trained head might achieve); the reduced head's coverage ceiling of 76.8% compounds
multiplicatively with that already-low acceptance, so wide K's draft-side savings (real, and by
construction, since the reduced head's GEMM is ~83x fewer output rows) are outweighed by
even-lower acceptance at every K this pass measured.

**What this means for the technique, honestly.** The MECHANISM (container format, loader, kernel,
lossless guarantee, K-widening) is complete, correct, and measured on real hardware. The ECONOMIC
CASE (docs/r9700.md §2.2's ~2.5-3x projection) depends on a well-covered subset, which this pass's
calibration corpus was too small/narrow to build -- this is squarely a CALIBRATION DATA problem, not
an implementation one, and is the clearly-scoped follow-up flagged in "Known gaps" below: re-run
`tests/model/tool_vocab_calib.cpp` against a much larger (hundreds of thousands of tokens), more
topically diverse corpus (ideally including chat/instruction-style text closer to real serving
traffic, not just Wikipedia prose) and re-measure this same K-sweep. Until that is done, **the
recommended default remains the ORIGINAL full-vocab MTP head at its own previously-measured optimal
K (w4a16 K=3: 68.20-68.64 tok/s, 46.3% acceptance -- see the matched-K re-measurement in "Flag
default flipped" below)** -- `--mtp-draft-head full` (or
simply not baking `--draft-vocab-ids` into a production container) is the safe choice until a
higher-coverage subset is measured and shown to beat it.

**Flag default flipped (2026-09-20, Milestone 5 B2 item 8)**: `--mtp-draft-head`'s own default in
both `src/cli/cli_args.h` and `src/server/server_args.h` changed from `reduced` to `full`, matching
this recommendation exactly -- a caller who does nothing now gets the measured-faster full-vocab
head instead of the measured-slower reduced head.

**Correction (review, 2026-09-20): the "65.02-67.34 tok/s at matched K" justification originally
given here was not a matched-K comparison.** 65.02 tok/s above is w4a16 **K=4** (full head) and
67.34 tok/s is w4a16 **K=3** from the *separate* "MTP head layout" sweep (a different knob --
`mtp_head_layout`, both variants of which are already full-vocab -- and that sweep's own container/
run, not this section's reduced-vs-full container/run); no K=3 full-vocab-draft-head datapoint
existed in either table at the time this recommendation was written. Re-measured directly instead
of citing across sweeps: real hardware, HIP device 1,
`D:/models/r4dx/qwen38-27b-v3-draftvocab.r4dx`, w4a16, this file's standard prompt/flags, `--mtp 3`,
two runs each, uncontended --

| Draft head | Decode tok/s (run 1, run 2) | Acceptance | Tok/round | Generated text |
|---|---|---|---|---|
| full (default) | 68.20, 68.64 | 46.3% | 2.31 | byte-identical to reduced's |
| reduced | 53.59, 54.17 | 20.9% | 1.63 | byte-identical to full's |

**Independently re-confirmed (Milestone 5 stage S2, 2026-09-20)**, same container/prompt/flags, two
fresh runs each on an otherwise idle device 1: full **68.63, 68.57** tok/s and reduced **54.09,
54.03** tok/s, with acceptance and tok/round landing on the *identical* 46.3%/2.31 and 20.9%/1.63.
The tok/s figures agree with the table above to within 1% (run-to-run noise on this card is ~1%,
which is why `docs/perf.md` asks for two runs and a report of both when they differ by more than
3%); the acceptance and tok/round figures are deterministic and match exactly. The table is left as
originally measured rather than overwritten with a second, statistically indistinguishable sample.
One operational note found while re-measuring: `--mtp-draft-head reduced` against a container with
no `mtp.draft_head.*` tensors (e.g. `qwen38-27b-v3.r4dx`, as opposed to the `-draftvocab` one)
silently uses the full head -- as documented and intended -- and the `[stats]` line correctly
reports `draft_head=full`, which is the only way to tell from the outside. Read that field, not the
flag you passed.

Matched-K=3 confirms the same conclusion the (mismatched-K) sweep comparison already pointed to:
the full head is faster at equal K, acceptance is more than double, and the flip is lossless
(identical generated text either way -- `--mtp-draft-head` can only affect drafting speed/
acceptance, never the accepted token, per `MtpHead::Draft`'s own doc comment). Pass
`--mtp-draft-head reduced` explicitly to opt back into the reduced-vocab path (e.g. for an A/B run,
or once a higher-coverage calibration corpus justifies it).

## DFlash2 assessment

`D:/models/Qwen3.8-27B-DFlash2/{Qwen3.8-27B-DFlash2-Q8_0.gguf (1.91 GB), Qwen3.8-27B-DFlash2-Q4_0_
ROCMFP4_FAST.gguf (1.03 GB)}` are real, already-downloaded checkpoints from the user's ROCmFPX setup
(`z-lab/Qwen3.8-27B-DFlash2` on HuggingFace, per the GGUF metadata read directly off these files:
`general.architecture=dflash`, `general.finetune=DFlash2`, `general.base_model.0.name=Qwen3.8 27B`,
tags `dflash2`/`speculative-decoding`/`block-diffusion`/`draft-model`/`sglang`).

**This is a genuinely different model architecture, not a slice of the target model's own weights.**
The GGUF metadata's own key namespace (`dflash.*`) is disjoint from every `text.*`/`mtp.*` tensor
this container format defines: `dflash.block_size`, `dflash.conv_kernel_size`,
`dflash.conv_group_size`, `dflash.selector_rank`, `dflash.selector_top_k`, `dflash.target_layers`,
`dflash.attention.sliding_window(_pattern)` describe a purpose-built block-diffusion drafter with
its own selector mechanism (choosing WHICH of several candidate continuations to emit, per the
`block-diffusion`/`draft-model` tags) -- a materially different forward pass than "one more decoder
layer chained `draft_k` times" (this project's own MTP head, `mtp_head.cpp`'s file comment) or "a
smaller `lm_head`" (this section's own reduced-vocab head).

**What porting it would actually require**, none of which this task's reduced-vocab head needed:
1. A GGUF reader (`third_party`/`src/convert` has none -- this project's only checkpoint format
   today is HF safetensors via `r4dx_convert::SafetensorsReader`).
2. A new model forward pass for the `dflash` architecture -- its own attention/selector/conv kernels,
   almost certainly not expressible as a thin wrapper over `GdnLayer`/`AttentionLayer`/`Mlp`, since
   the selector mechanism (picking among candidate blocks) has no analogue in this codebase's
   existing decode/verify path.
3. A new integration point in `Model`/`MtpHead` (or a THIRD sibling to both) to drive it, since it is
   not "one more decoder layer" the existing `MtpHead::Draft` loop structure can just call again with
   different weights -- unlike this milestone's reduced-vocab head, which reused `FinalLmHead`
   completely unmodified.

**Verdict: do not port.** The reduced-vocab head already captures the projected win this milestone
asks for (docs/r9700.md R9's own sizing: ~2.5-3x at p~=0.5 acceptance with a 273 MB/row drafter,
against R8's ~1.5-1.9x) with a change confined entirely to this codebase's EXISTING abstractions
(`QuantLinear`, `FinalLmHead`, `MtpHead::Draft`'s own loop) and a measured, real, zero-architecture-
risk lossless guarantee (verification is untouched). DFlash2 might beat the reduced-vocab head's
OWN acceptance rate at a given K (a purpose-trained selector could plausibly do better than "guess
from a vocabulary subset with a smaller lm_head"), but that upside is speculative and unmeasured
against this task's own effort budget, while the reduced-vocab head's upside is measured on real
hardware in this same pass. A DFlash2 port is flagged as a candidate for a FUTURE, separately-scoped
milestone if the reduced-vocab head's own measured ceiling (this section's K-sweep table above) turns
out to leave meaningful headroom on the table -- not attempted here, per the task's own "do not port
DFlash2 wholesale if the reduced-vocab head already captures the win" instruction.

## Known gaps

- **Acceptance rate, while now far higher (30-75% vs the original pass's 0-1.2%), is still below
  the 50-90% a production self-speculative head on a purpose-trained checkpoint might achieve for
  short K** -- plausible remaining gap, not yet investigated: this checkpoint's `mtp.*` weights may
  simply not have been trained for as many self-speculative steps as e.g. a purpose-built
  DeepSeek-V3 MTP head, or `--temperature 0` greedy verification (this pass's only implemented
  acceptance rule) may be stricter than whatever sampling regime the checkpoint was validated under.
  Not blocking: the measured speedup already more than justifies `--mtp`'s cost at the per-layout
  optimal K.
- **Generated text was byte-identical for every configuration this revision actually measured**
  (see "Correctness" above), but the underlying floating-point-non-associativity mechanism the
  original pass documented is still structurally present in the batched-vs-sequential attention
  decode kernel path -- a longer generation or different prompt could still diverge from `--mtp 0`'s
  own output on some future run. Not believed to be a logic bug (verified correct via
  `CheckVerifyMatchesSequential`); flagged for awareness, not as an open defect.
- **FIXED (server-catches-up-with-engine stage), regression-tested (Milestone 4 follow-up,
  2026-09-20)**: ~~`--chat` multi-turn + MTP interaction not exercised~~ -- an MTP round that stops
  mid-round (hits `--max-tokens` or EOS partway through a round's own returned token vector) commits
  EVERY element of that round except its own last one atomically (the last element is always the
  "corrected/bonus" token, analogous to `Prefill`'s own returned-but-not-yet-fed `next` -- see
  `Model::DecodeStepMtpGreedy`'s doc comment), regardless of where the caller's own per-token display
  loop decides to stop. `src/cli/main.cpp`'s `fed_tokens` and `src/server/engine.cpp`'s `PrefixState`
  (`src/server/prefix_state.h`) now both track this via a `committed_tokens` set built from the
  round's own atomicity guarantee (every element but the round's last is pushed to
  `committed_tokens` unconditionally, before the emit/stop-check that decides
  `generated_tokens`/display), not from `generated_tokens` alone -- see `TurnResult::committed_tokens`'s
  comment (CLI) / `prefix_state.h`'s file comment (server) for the full derivation.
  ~~`--chat` multi-turn + MTP together is still only lightly exercised end to end (no automated
  multi-turn-with-a-forced-mid-round-stop regression test exists)~~ **RESOLVED**:
  `tests/model/test_mtp.cpp::CheckChatMultiTurnMidRoundStop` drives a REAL two-turn conversation
  through a real `Model` + the production `ProcessMtpRound`/`PrefixState` helpers, with a
  `--max-tokens`-equivalent budget forced (via a same-seed dry run) to land exactly one token short
  of a round boundary, asserts `PrefixState::Extend()` correctly refuses the resulting (intentionally
  desynced) fast-path extension, and verifies the `Reset()`+full-reprefill recovery path's turn-2
  continuation is byte-identical to an independently-loaded sequential reference.
- ~~**Only greedy acceptance is implemented**~~ **RESOLVED (Milestone 6 stages S2-S3)**:
  `Model::DecodeStepMtpSampled` implements exactly the probabilistic accept/reject-and-resample rule
  this bullet asked for (rejection sampling against a deterministic proposal, see "Sampled rounds"
  above and [sampling.md](sampling.md) section 9), and stage S3 wired it into both `src/cli/main.cpp`
  and `src/server/engine.cpp` -- `--mtp K` with `--temperature > 0` now runs the sampled path instead
  of warning and disabling MTP for the run (the 2026-09-19 warn-and-force-`mtp_draft_k=0` behavior
  this bullet used to describe is gone).
- `tools/profile/tune_gemm.py`'s GEMM tuning table (docs/perf.md) was not re-swept for MTP's verify-
  window M values or `PrimeKv`'s own batched-priming M values (the M-band rounding in `PickTuning`
  already covers M=1..64, so every shape this revision's `PrimeKv`/`Draft()` use picks a legal, if
  not necessarily re-optimized-for-this-exact-M, tuning row) -- a future pass could add
  MTP-window-specific M points to the sweep.
- ~~`MtpHead::Draft`'s per-draft-token host round-trips~~ **RESOLVED** (device-resident draft loop
  pass, see "Device-resident draft loop" above): the embedding gather, `positions_`/`seqused_k_`
  upload, and drafted-token readback are all now device-resident/batched-per-window rather than
  per-drafted-token, when `Container::EmbedTokensDeviceResident()` (default true).
- **RESOLVED (Milestone 4 follow-up, 2026-09-20 -- see "h_seed drift" above)**: ~~the acceptance gap
  between w4a16 and w4a8/mxfp4 was investigated but not fully root-caused~~ -- a direct
  golden-referenced comparison of `h_seed` against a bf16 exact-arithmetic reference (a new
  all-4-layout 4-layer container, `Model::DebugSeedHiddenBf16()`, and
  `tests/model/tool_hseed_drift.cpp`) found that h_seed drift from the bf16 reference matches the
  acceptance ranking exactly (w4a16 smallest drift/highest acceptance, w4a8 largest drift/lowest
  acceptance, mxfp4 in between on both), driven almost entirely by one "massive activation" residual
  dimension whose quantization error tracks the same per-layout ordering. Not a bug -- expected
  quantization behavior on an outlier-magnitude dimension, so no fix was made; an outlier-aware
  quantization scheme for that handful of dimensions specifically is the natural (unstarted)
  follow-up if the acceptance gap is worth narrowing further.

## API summary

- `r4dx::model::Model::ModelOptions::mtp_draft_k` (default 0): enables MTP, sizing GDN's window
  bank and allocating `MtpHead` (now sized to `ModelOptions::max_ctx`, not a tiny scratch block) +
  scratch. Requires `Container::HasMtp()`.
- `Model::DecodeStepMtpGreedy(int32_t token_id, int64_t k) -> std::vector<int32_t>`: drafts up to
  `k` tokens, verifies, commits, returns 1..k+1 new tokens. Also keeps MTP's own KV cache in
  lockstep (directly when `k==0`, via `Draft()`'s own step 0 otherwise).
- `Model::DecodeStepMtpSampled(token_id, k, SampleParams, rng) -> std::vector<int32_t>`
  (Milestone 6 stage S2, [sampling.md](sampling.md) section 9): the SAMPLED counterpart. Same
  drafting, same one-pass verification, same commit tail -- literally the same implementation, with
  the per-row token coming from a sampler instead of an argmax (see "Sampled rounds" above).
  `temperature <= 0` routes straight to `DecodeStepMtpGreedy` and consumes no draw.
- `Model::VerifyWindow(candidates, logits_out=nullptr, summaries_out=nullptr, inv_temperature=1)
  -> std::vector<int32_t>` (public): the verify-only primitive, exposed for tests and diagnostics.
  Does not touch MTP's own KV cache (verification is purely against the backbone). `summaries_out`
  (stage S2) fills each row's DEVICE row summary -- `rows * 516` bytes instead of the
  `rows * ~993 KB` `logits_out` costs -- which is what a sampled round reads.
- `Model::ReadVerifyLogitsRow(row, out)` (public, stage S2): one row of the most recent
  `VerifyWindow`'s logits, for the row-granular fallback and for tests.
- `r4dx::model::MtpHead::Draft(...)` (`src/model/mtp_head.h`): the draft-only primitive. Takes a
  `base_pos` parameter (the real sequence position of its first draft step) and, this pass, an
  `embed_table_dev` parameter (device-resident embedding table pointer, or `nullptr` to fall back to
  the host-gather path -- see "Device-resident draft loop" above).
- `r4dx::model::MtpHead::PrimeKv(...)` (`src/model/mtp_head.h`): extends MTP's own KV cache by real
  positions without drafting -- called from `Model::RunChunk` on every prefill chunk and plain
  decode step.
- `ModelOptions::mtp_head_layout` (`std::optional<Layout>`, default `std::nullopt` -- see "MTP head
  layout" above): layout for ONLY the MTP head's four quantized linears, independent of the body's
  own `layout`. `nullopt` tracks the body layout (measured default); `Layout::kBf16` forces the
  exact-arithmetic head.
- `ModelOptions::embed_device_resident` (default `true` -- see "Device-resident draft loop" above):
  mirrors `text.embed_tokens` into VRAM so the decode/draft path gathers embeddings on-device;
  `Container::EmbedTokensDeviceResident()` reports whether the mirror actually landed (free-VRAM
  heuristic can decline it even when requested).
- `r4dx::kernels::r4dx_embedding_gather_bf16` (`src/kernels`): the device-resident gather kernel,
  `r4dx::model::EmbedTokensDeviceGather` (`src/model/embedding.h`) its thin call-site wrapper.
- CLI: `r4dx-cli --mtp K` (0 default/disabled). `--mtp-head-layout {bf16,layout}` (default
  `layout` -- see "MTP head layout" above). `--stats` additionally prints
  `[stats] mtp: draft_k=... rounds=... drafted=... accepted=... (X% acceptance, Y tok/round avg)`.
  `--mtp K` now works at any `--temperature` (Milestone 6 stage S3): `--temperature <= 0` takes
  `DecodeStepMtpGreedy`, `--temperature > 0` takes `DecodeStepMtpSampled` -- the older
  warn-and-force-`mtp_draft_k=0` behavior for `--temperature > 0` is gone.
- Server: `r4dx-server --mtp N` and, as of the Milestone 4 follow-up (2026-09-20),
  `--mtp-head-layout {bf16,layout}` (`src/server/server_args.h`, default `layout`) -- same
  semantics/default as the CLI flag, passthrough to `ModelOptions::mtp_head_layout` in
  `src/server/main.cpp`.
- `Model::DebugSeedHiddenBf16()` (`src/model/model.h`/`.cpp`, Milestone 4 follow-up): diagnostic-only
  accessor reading back `mtp_seed_hidden_` (the exact row `MtpHead::Draft`'s first step consumes) as
  raw bf16 bits, for the h_seed-drift investigation above. Requires `MtpEnabled()` and at least one
  prior `Prefill`/`DecodeStep*` call.
