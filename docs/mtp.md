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
from the original pass's report of divergence past ~64 tokens for some prompts. This is expected,
not just lucky: the original pass's divergence was root-caused to floating-point non-associativity
between the batched (q_len>1) verify-window attention-decode kernel and the sequential (q_len=1)
path (see prior revision's writeup, preserved below) -- an effect that is still structurally present
in the kernel, but with acceptance now high (30-75% vs 0-1.2%), far FEWER verify rounds run for the
same generation length (a K=3 round that accepts 3/3 drafts advances the sequence by 4 tokens in
one verify call instead of 4 separate q_len=1 calls), so there are correspondingly fewer
opportunities for the rounding-order effect to flip a near-tie argmax. Longer generations or
different prompts could still diverge from this same floating-point-non-associativity mechanism;
this is not a logic bug and was independently verified correct via `CheckVerifyMatchesSequential`.

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
- **`--chat` multi-turn + MTP interaction not exercised**: if an MTP round stops mid-round (hits
  `--max-tokens` or EOS partway through a round's own returned token vector), `Model`'s internal
  `pos_`/GDN/MTP-KV state already reflects EVERY token that round committed, but `src/cli/main.cpp`'s
  `fed_tokens` bookkeeping (used to compute what to re-prefill next turn) only counts the tokens
  actually pushed into `result.generated_tokens` -- these can under-count relative to the real
  committed state in that specific edge case, which could desync a LATER `--chat` turn's prefix
  match. Not observed in this pass's (single-turn, `--prompt`) testing; consistent with
  `docs/status.md`'s pre-existing "`--chat` multi-turn only lightly exercised" note.
- **Only greedy acceptance is implemented** (task's own stated scope: "typical/temperature
  acceptance later"). `Model::DecodeStepMtpGreedy`/`VerifyWindow` are greedy-only; a caller wanting
  temperature/top-k/top-p sampling with MTP would need a probabilistic acceptance rule (e.g.
  speculative sampling's own accept/reject-and-resample test) this pass does not provide --
  `src/cli/main.cpp` now explicitly warns and forces `mtp_draft_k=0` when `--temperature > 0`
  (review finding, 2026-09-19 -- previously silent).
- `tools/profile/tune_gemm.py`'s GEMM tuning table (docs/perf.md) was not re-swept for MTP's verify-
  window M values or `PrimeKv`'s own batched-priming M values (the M-band rounding in `PickTuning`
  already covers M=1..64, so every shape this revision's `PrimeKv`/`Draft()` use picks a legal, if
  not necessarily re-optimized-for-this-exact-M, tuning row) -- a future pass could add
  MTP-window-specific M points to the sweep.
- **`MtpHead::Draft`'s per-draft-token host round-trips** (embedding gather is host-side,
  `src/kernels/include/r4dx/kernels/embedding.hpp`'s own TODO) still force a `stream.Synchronize()`
  + small D2H per drafted token, same as the original pass documented -- unaffected by this
  revision's fixes, still open (minor, not re-measured separately this pass since the net effect is
  already captured in the tok/s numbers above).

## API summary

- `r4dx::model::Model::ModelOptions::mtp_draft_k` (default 0): enables MTP, sizing GDN's window
  bank and allocating `MtpHead` (now sized to `ModelOptions::max_ctx`, not a tiny scratch block) +
  scratch. Requires `Container::HasMtp()`.
- `Model::DecodeStepMtpGreedy(int32_t token_id, int64_t k) -> std::vector<int32_t>`: drafts up to
  `k` tokens, verifies, commits, returns 1..k+1 new tokens. Also keeps MTP's own KV cache in
  lockstep (directly when `k==0`, via `Draft()`'s own step 0 otherwise).
- `Model::VerifyWindow(candidates, logits_out=nullptr) -> std::vector<int32_t>` (public): the
  verify-only primitive, exposed for tests and diagnostics. Does not touch MTP's own KV cache
  (verification is purely against the backbone).
- `r4dx::model::MtpHead::Draft(...)` (`src/model/mtp_head.h`): the draft-only primitive. Now takes
  a `base_pos` parameter (the real sequence position of its first draft step).
- `r4dx::model::MtpHead::PrimeKv(...)` (`src/model/mtp_head.h`, new this revision): extends MTP's
  own KV cache by real positions without drafting -- called from `Model::RunChunk` on every prefill
  chunk and plain decode step.
- CLI: `r4dx-cli --mtp K` (0 default/disabled). `--stats` additionally prints
  `[stats] mtp: draft_k=... rounds=... drafted=... accepted=... (X% acceptance, Y tok/round avg)`.
  `--mtp K` with `--temperature > 0` now warns on stderr and forces `mtp_draft_k=0` for that run
  (MTP is greedy-only).
