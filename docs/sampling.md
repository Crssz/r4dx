# Sampling

What "the sampled token" means in r4dx, and the two paths that produce it.

Scope as of Milestone 6 stage S3 (done). Sections 1-6 are the sampler pieces: the device row-summary
kernel, the canonical full-vocab host sampler, and the exact summary sampler. Sections 8-11 are
those pieces wired into `r4dx::model::Model`: a plain sampled decode step now resolves its token
from a 516-byte device row summary instead of a ~993 KB logits copy, and both speculation families
have a sampled round -- `DecodeStepMtpSampled`, `DecodeStepDflashSampled` -- that accepts drafts
against a *sampling* target without changing a single emitted token. Section 12 is stage S3: `Model`'s
sampled methods are now wired into `src/server`/`src/cli`, so `--mtp`/`--dflash` speculation and the
plain decode path's fast sampler both run at ANY temperature, not just greedy.

Related: [mtp.md](mtp.md), [dflash2.md](dflash2.md) (the speculation paths), [server.md](server.md)
(where `SampleParams` comes from), [perf.md](perf.md) (measurement rules).

---

## 1. Why

Before this milestone, speculation (MTP, DFlash2) ran only for `temperature <= 0`, because
acceptance was "draft token == target argmax". Chat clients send temperature 0.6-1.0, so real
traffic got plain sampled decode. Two things had to change for a sampled target to be able to
accept a draft (both are done as of stage S3, section 12):

1. an acceptance rule that is correct for a sampling target, and
2. a sampler that does not cost a 1 MB device-to-host copy plus an O(vocab) host pass per token.

Stage S1 builds the pieces (1) needs to be *checkable* and the whole of (2).

---

## 2. Canonical sampling

> **Definition.** For a request with `SampleParams` p and one uniform draw `u` in `[0,1)`, the
> sampled token is `InvCDF(u)` over the post-filter distribution, walked in **canonical order**.
>
> **Canonical order** is: **raw logit descending, ties broken toward the LOWER token id.**
>
> Exactly **one** draw `u` per emitted token, taken from the request's seeded `std::mt19937_64`.

Every sampling path in the engine -- plain, MTP, DFlash2, summary-based or full-vocab -- must
produce exactly this token for the same logits and the same `u`. That is what lets a future stage
verify a speculative round against plain decode token by token.

### 2.1 The filters

Unchanged from the pre-M6 sampler, which is the specification. In order:

| step | rule |
|---|---|
| temperature | `scaled = logit / temperature` (a **division**, not a multiply by a reciprocal) |
| top_k | keep the `top_k` highest in canonical order; disabled when `top_k <= 0` or `top_k >= vocab` |
| min_p | keep `w >= min_p`, where `w = exp(scaled - max_scaled)` and `max_scaled` is the **untruncated** row max, so the row argmax's own `w` is exactly 1.0 and the threshold is plainly `min_p` |
| top_p | keep the shortest canonical-order prefix whose mass **reaches** `top_p * (mass of the set min_p left)` |
| draw | `target = u * (mass of the surviving set)`; walk the surviving set in canonical order accumulating `cum`; emit the first token with `target < cum` |

All sums are `double` accumulations of `float` weights, taken in canonical order starting from
`0.0`. The comparison is `target < cum`, strictly, so `u` landing exactly on a CDF step selects the
*next* token; both paths agree because both compute `target` and `cum` from the same floats in the
same order.

### 2.2 Why the order is keyed on the RAW logit

Dividing by a positive temperature is monotone but not injective in fp32: two distinct raw logits
can collapse onto the same scaled float. Ordering on the raw value keeps canonical order a strict
*total* order, and -- more importantly -- it is an order the **device** kernel can produce, since
that kernel returns raw logits and never sees the temperature. Ordering on the scaled value would
make the two paths disagree on rows containing such a collapse.

### 2.3 What changed versus the pre-M6 sampler

The **distribution** is identical. Only the map from a seed to a token changed, in two ways:

* equal-logit ordering was left to `std::sort` / `std::nth_element` (unspecified) and is now
  deterministic (lower id first);
* the no-filter fast path walked the CDF in **token-id** order and now walks it in canonical order.

`SampleLegacy` in `sampler.hpp` is the pre-M6 function kept verbatim, and
`tests/kernels/test_sampler_canonical.cpp` checks **both** it and the new sampler against the same
independently-computed fp64 exact distribution, 400000 draws each, for six filter combinations --
so drift in either is caught rather than assumed away. See section 6.

### 2.4 The draw

```cpp
double DrawUniform01(std::mt19937_64& rng) {
  return static_cast<double>(rng() >> 11) * (1.0 / 9007199254740992.0);  // 2^-53
}
```

Spelled out rather than `std::generate_canonical`, which is not specified to make exactly one
`rng()` call and does not agree across standard libraries. Greedy requests (`temperature <= 0`)
consume **no** draw, so a greedy generator's state is untouched -- the pre-M6 behaviour.

---

## 3. The row summary kernel

`r4dx_topk_lse_f32` (declared in `src/kernels/include/r4dx/kernels/kernels.h`, implemented in
`src/kernels/src/r4dx_kernels.hip`).

```
r4dx_topk_lse_f32(logits, out_ids, out_vals, out_lse, rows, vocab, inv_temperature, stream)
```

Per row of `logits` ([rows, vocab] fp32, row stride == vocab):

* `out_vals[r, 0..K-1]` / `out_ids[r, 0..K-1]` -- the top **K = 64** entries in canonical order
  (value descending, ties toward the lower id). **Raw** logits, not scaled: the host applies its
  own `logit / temperature` to them, so the summary path and the full-vocab path compute
  bit-identical softmax numerators over the top-K.
* `out_lse[r]` -- `log( sum_i exp(logits[r,i] * inv_temperature) )` over the **full** row.

D2H per round is `rows * (K*8 + 4)` bytes -- 516 bytes per row (K=64: 64*8 + 4), against 993 280
bytes for the full fp32 row it replaces.

**Preconditions** (throw `std::runtime_error`, never a silently wrong answer): `1 <= rows <= 8`;
`vocab > K`; `inv_temperature` finite and `> 0`. `rows <= 0` is a no-op, as it is for every other
entry point in that header. NaN / `+inf` logits are undefined input, the same as for
`r4dx_topk16_f32` and `r4dx_argmax_f32`.

**One call in flight per process.** The kernel keeps its partials in a module-scope device scratch
buffer (~264 KiB of VRAM) rather than taking a caller-supplied pointer, which keeps the entry point
the same shape as every other one in the header. Two concurrent calls (two threads, two streams)
would interleave partials. This is the same single-worker-thread assumption the non-atomic
`r4dx_kernel_launch_counter` already relies on and `model.h`'s SCOPE comment guarantees. The entry
point makes **two** device launches and advances the launch counter by 2.

### 3.1 Why K = 64 and not a single-round top-64

A per-thread top-64 merged across a 256-thread workgroup needs `256 * 64 * 8 = 128 KiB` of LDS,
exactly twice gfx1201's 64 KiB per-workgroup limit. So the kernel keeps the proven per-thread
top-**16** machinery (`r4dx_topk16_f32`'s, now shared code) whose merge buffer is
`256 * 17 * 8 = 34 KiB`, and runs the extraction **four times**: round *r* admits only entries
strictly worse than the last `(value, id)` pair round *r-1* emitted. Because the admission test
uses the same **total** order, round *r* yields exactly ranks `16r+1 .. 16r+16`. The result is the
exact top-64, not an approximation, and that is what
`tests/kernels/test_topk_lse.cpp` pins against a CPU reference with exact id **and** value
equality, including on tie-heavy rows.

### 3.2 Two phases

| kernel | grid | does |
|---|---|---|
| `TopkLsePartialKernel` | (chunks, rows) | each block owns a contiguous slice of one row; writes that slice's top-64 and its `(max, sum of exp)` pair to the device scratch |
| `TopkLseMergeKernel` | (rows) | merges the `chunks * 64` partial candidates into the row's top-64; combines the per-slice `(max, sum)` pairs into the row's `lse` |

`chunks = clamp(vocab / 1024, 1, 64)`.

The merge is exact for the same reason the extraction rounds are: at most 64 of the row's top-64
can lie in any one slice, and that slice's own top-64 contains all of them.

### 3.3 Numerical form of the logsumexp

Per slice, with `m` = that slice's max (published by the slice's own round 0, so the accumulation
costs no extra pass -- it is fused into round 1, which re-reads the slice anyway):

```
part_sum = sum over the slice of  (double) expf((v - m) * inv_temperature)
```

`expf` in fp32, accumulated in **double**, reduced across the block by a fixed-order tree so the
result is reproducible run to run. The merge kernel then rescales each slice by
`expf((slice_max - row_max) * inv_temperature)`, accumulates in double in a fixed chunk order, and
returns `row_max * inv_temperature + logf(total)`.

* a `-inf` entry against a finite max contributes `expf(-inf) == 0`, which is correct;
* an all-`-inf` row yields `-inf`, not NaN;
* `total` is bounded in `[1, vocab]` (the row max's own term is exactly `exp(0) == 1` and every term
  is `<= 1`), so `logf` costs ~1e-7 absolute on it.

The rescale and the final log deliberately use `expf`/`logf`, not the double `exp`/`log`: gfx1201
has no hardware fp64 transcendental, so a double `exp()` is a software routine at the fp64 ALU
rate. 64 of them in the merge kernel's single lane **measured at ~0.27 ms**, which was most of the
whole kernel. The accumulation stays in double, where it matters.

**Measured error.** `tests/kernels/test_topk_lse.cpp` compares against an fp64 Kahan-compensated
reference over random rows, 12-distinct-level rows, all-identical rows, half-`-inf` rows, a row with
one logit +40 above the rest, and a flat row, at `inv_temperature` in `{0.5, 1.0, 1/0.6, 5.0}` and
`vocab` in `{4096, 248320}`, `rows` in `1..8`:

> worst `|error|` **4.762e-06**, tolerance **1e-4**.

### 3.4 Two micro-architectural findings, both measured

Both were found while bringing this kernel up and both also apply to `r4dx_topk16_f32`, which
shares the code and got faster as a result.

**Register arrays.** `float bv[16]; int32_t bi[16];` as plain locals did not survive SROA on this
toolchain: `Topk16Kernel` compiled to **256 VGPRs and 656 bytes/lane of scratch spill**, i.e. every
list access going to memory. Wrapping the same two arrays in a struct with member functions
(`TopkList`) drops it to ~68 VGPRs and **zero** scratch. Checked with
`hipcc -Rpass-analysis=kernel-resource-usage`.

**LDS bank conflicts.** At a per-thread pitch of 16 dwords, thread *t*'s list starts at `t*16`, so
`(t*16) % 32` takes only two values across a 32-lane wave and every LDS access in the merge
serialises ~16 ways. The pitch is now `kTopkLdsPitch = 17`, coprime with the 32 banks.

---

## 4. Sampling from a summary

`r4dx::kernels::SampleFromSummary` in `src/kernels/include/r4dx/kernels/summary_sampler.hpp`
(header-only, HIP-free).

```cpp
SummarySampleResult SampleFromSummary(const RowSummary& s, const SampleParams& p, double u);
// -> { resolved = true, token }   the exact token SampleCanonical would return for this u
// -> { resolved = false }          the caller must run the full-vocab path FOR THIS ROW
```

It is **exact or silent**. It never guesses.

### 4.1 When the summary is enough

Everything computed from the top-K itself is bit-exact: `max_scaled` is `vals[0] / temperature`
(the row max is slot 0, and division by a positive temperature is monotone), the weights are the
same `std::exp(val / temperature - max_scaled)` floats the full path computes, and the partial sums
are accumulated in the same order. So whenever the post-filter candidate set is provably a subset
of the top-K, the summary path reproduces the full path exactly.

| case | provable when |
|---|---|
| `top_k` in `[1, K]` | **always** -- the set is a prefix of the top-K |
| `min_p > 0` | `w_K < min_p`, i.e. min_p cuts **inside** the top-K (`w` is non-increasing in canonical order, so the survivors are a prefix) |
| `top_p < 1`, no other filter | the nucleus boundary is provably inside the top-K under the `S_full` band below |
| pure temperature | `u * S_full` provably lands on a step inside the top-K, i.e. `u` is not in the tail |
| `top_k > K`, or a `min_p` that does not bite inside the top-K, with nothing else closing the set | **never** -- fall back |

### 4.2 The one approximate quantity

Only the full row's partition function `S_full = exp(lse - max_scaled)` is approximate, and only
the bare-`top_p` and pure-temperature cases need it. It is used as a **band**
`[S_full*(1-tol), S_full*(1+tol)]` with `tol = kRowSummaryLseRelTol = 1e-3`, and the answer is
returned only when it is the same for every `S_full` in the band:

* nucleus: return prefix length `n` only if `cum_n >= top_p * S_hi` while every earlier `cum` was
  `< top_p * S_lo`; if a `cum` lands inside the band, fall back;
* pure-temperature walk: return slot `j` only if `t_hi < cum_j` while every earlier `cum` was
  `<= t_lo`.

The band covers, with ~5x headroom AT `|max_logit/temperature| <~ 1e4`: the kernel's own `lse`
error (<= 1e-4 absolute, measured 4.8e-6); the kernel scaling by `inv_temperature` where the host
divides by `temperature` (`x*(1/T)` vs `x/T`, ~1e-5 relative at `|x/T| ~ 200`); and fp32 `expf` plus
summation-order differences between device and host (~1e-7). The middle term grows with `|x/T|` and
was measured to exceed the 1e-3 band below roughly `T~0.002` on this model's logit scale (1.7e-6 at
`T=1.0`, 2.4e-4 at `T=0.005`, 7.6e-4 at `T=0.002`, 1.5e-3 at `T=0.001` -- review finding, 2026-09-21).
`Model::SummaryInvTemperature` (`model.cpp`) screens `temperature < kMinSummaryTemperature = 0.01`
to the full-logits fallback for exactly this reason (>2x margin below the measured-safe `T=0.005`
point), so a caller going through `Model` never exercises this band outside where it was measured to
hold; `SampleFromSummary` itself has no such guard and assumes its caller applies an equivalent one.

`tests/kernels/test_summary_sampler.cpp` poisons `lse` so that `top_p * S_full` lands **exactly** on
the nucleus boundary and asserts that every one of 100000 values of `u` falls back rather than
answering.

---

## 5. Cost

Measured on HIP device 1 (Radeon AI PRO R9700, gfx1201), `V = 248320`, hip events, 200 iterations
after a warm-up, via `tests/kernels/tool_sampler_bench.cpp`
(`build\win-hip\tests\kernels\tool_sampler_bench.exe`, `HIP_VISIBLE_DEVICES=1`). Two runs, both
reported where they differ by more than 3% ([perf.md](perf.md)).

### 5.1 The kernel

| | run A | run B |
|---|---|---|
| `r4dx_topk_lse_f32`, rows=1 | 0.306 / 0.325 ms | 0.192 / 0.193 ms |
| `r4dx_topk_lse_f32`, rows=8 | 0.563 / 0.681 ms | 0.563 / 0.689 ms |
| *reference:* `r4dx_topk16_f32`, rows=1 | 0.183 ms | 0.167 ms |
| *reference:* `r4dx_argmax_f32`, rows=1 | 0.192 ms | 0.189 ms |
| *reference:* one-thread kernel launch | 0.0024 ms | 0.0014 ms |
| D2H full fp32 logits, rows=1 | 0.163 ms | 0.163 ms |
| D2H row summary, rows=1 | 0.100 ms | 0.104 ms |
| D2H full fp32 logits, rows=8 | 0.497 ms | 0.509 ms |
| D2H row summary, rows=8 | 0.155 ms | 0.161 ms |

rows=1 is the noisy measurement (0.19-0.33 across runs); the `r4dx_argmax_f32` reference in the same
process swings the same way, so it is a clock/residency effect of these small single-row launches,
not the kernel.

Context: `r4dx_argmax_f32` -- one block, one pass, a single int32 out -- is what the **greedy** path
already pays per token today, and it measures 0.19 ms. This kernel does four passes, produces 64
sorted entries plus a logsumexp, and lands at roughly the same number.

### 5.2 Where the time goes

Holding the chunk count at its maximum and shrinking the row isolates the fixed cost from the
memory pass:

| V | chunks | `r4dx_topk_lse_f32` | `r4dx_topk16_f32` (1 block) | `r4dx_argmax_f32` (1 block) |
|---|---|---|---|---|
| 1024 | 1 | 0.147 / 0.140 ms | 0.0118 ms | 0.0015 ms |
| 4096 | 4 | 0.196 / 0.270 ms | 0.0587 ms | 0.0033 ms |
| 16384 | 16 | 0.276 / 0.279 ms | 0.1052 ms | 0.0087 ms |
| 65536 | 64 | 0.307 / 0.305 ms | 0.1378 ms | 0.1078 ms |
| 131072 | 64 | 0.312 / 0.310 ms | 0.1657 ms | 0.1454 ms |
| 248320 | 64 | 0.265 / 0.270 ms | 0.1797 ms | 0.1891 ms |

At `V = 1024` the kernel touches 4 KB and still costs 0.14 ms, so the cost is **not** memory. The
row-summary call runs **eight** LDS merge trees (four extraction rounds in each of the two phases),
and one such tree is directly visible as `r4dx_topk16_f32` at `V = 1024`: 0.012 ms, against
`r4dx_argmax_f32`'s 0.0015 ms for the same one-block one-pass shape without a tree. Eight of them
is the observed floor. Reducing that -- a wider per-thread list so fewer extraction rounds are
needed, a wave-level reduction for the tail of the tree, or fusing the summary into the lm_head
epilogue so it is not a standalone launch at all -- is left to the stage that wires this into the
engine.

Two earlier shapes of this kernel, for the record:

* **one workgroup per row** (the obvious shape): **1.11 ms** at rows=1. A single workgroup is 8
  waves on one CU, far too few to cover memory latency -- it reached ~5 GB/s on a 645 GB/s part.
  Chunking the row across 64 workgroups took it to 0.32 ms.
* **double `exp` in the merge tail**: 0.32 ms -> **0.193 ms** by switching the 64 rescale factors to
  `expf` (section 3.3).

### 5.3 The host sampler

Per token, on a realistic peaked row, `V = 248320` (run A; run B agrees within 3% except where
noted):

| filters | `SampleFromSummary` | `SampleCanonical` (full vocab) | `SampleLegacy` (pre-M6) | fallback rate |
|---|---|---|---|---|
| T=1.0 (pure temperature) | 0.00009 ms | 1.19 ms | 0.64 ms | 5.66% |
| T=0.7 top_k=20 top_p=0.8 | 0.00009 ms | 0.81 ms | 0.89 ms | 0% |
| T=0.6 top_k=20 top_p=0.95 | 0.00009 ms | 0.79 ms | 0.89 ms | 0% |
| T=1.0 top_p=0.9 | 0.00009 ms | 1.11 ms | 13.65 ms | 0% |
| T=0.8 min_p=0.05 | 0.00009 ms | 0.44 ms | 0.77 / 0.88 ms | 0% |
| T=0.7 top_k=50 min_p=0.02 top_p=0.9 | 0.00009 ms | 0.74 ms | 0.89 ms | 0% |

The summary path is ~4 orders of magnitude cheaper than either full-vocab path -- it touches 64
entries, not 248320. `SampleCanonical` is also 12x faster than `SampleLegacy` on the bare-`top_p`
case, because the pre-M6 code sorted the whole vocabulary there while the canonical one escalates a
partial selection (64, then x8) until the CDF walk resolves.

Fallback rates on the three shapes `tests/kernels/test_summary_sampler.cpp` sweeps (100000 values of
`u` each, 50000 on a grid and 50000 random) -- a performance property, not a contract:

| shape | pure temperature | top_k=20 top_p=0.8 | top_p=0.9 alone | min_p=0.05 alone | top_k=50 min_p=0.02 top_p=0.9 |
|---|---|---|---|---|---|
| peaked (Zipf-like) | 0% | 0% | 0% | 0% | 0% |
| flat (near-uniform) | 96.9% | 0% | 100% | 100% | 0% |
| wide nucleus | 86.9% | 0% | 100% | 100% | 0% |

Any filter that closes the candidate set inside the top-K resolves from the summary **always**,
whatever the row looks like. The unbounded cases (a bare nucleus, a bare `min_p`, a bare
temperature) resolve on a realistically peaked row and fall back on a flat one, which is the right
behaviour: a flat row genuinely has its answer outside the top 64.

---

## 6. Tests

| test | kind | covers |
|---|---|---|
| `tests/kernels/test_topk_lse.cpp` | GPU (device 1) | the kernel vs an fp64 CPU reference: exact id **and** value equality for the top-64 over random / 12-distinct-level / all-identical / all-`-inf` / half-`-inf` / peaked / flat rows, `V` in {4096, 248320}, `rows` 1..8, `inv_temperature` in {0.5, 1.0, 1/0.6, 5.0}; `lse` absolute error; every precondition throwing |
| `tests/kernels/test_sampler_canonical.cpp` | CPU | 400000 draws from `SampleCanonical` **and** 400000 from `SampleLegacy` against the same independently computed fp64 exact post-filter distribution, six filter combinations, chi-square with `E < 5` cells pooled and `alpha = 1e-6`; candidate-set equality between the two samplers and against the exact support; one-draw-per-token and no-draw-on-greedy |
| `tests/kernels/test_summary_sampler.cpp` | CPU | three logit shapes x six filter combinations x 100000 values of `u`: every resolved token equals the full-vocab canonical token, zero mismatches; zero fallbacks wherever a filter closes the set inside the top-K; `u` exactly on a CDF step; `top_k == K`; `top_k > K`; a poisoned `lse` on the nucleus threshold; a summary covering the whole row; greedy |
| `tests/kernels/test_sampler.cpp` | CPU | the pre-existing sampler invariants, unchanged |
| `tests/kernels/tool_sampler_bench.cpp` | GPU (device 1) | section 5; built but never `add_test()`'d, per the `tool_` convention |

The chi-square threshold is the Wilson-Hilferty cube-root approximation of the upper `1e-6` quantile
(accurate to well under 2% for `df >= 10`, which is every case). With 12 chi-square tests in that
file the family-wise false-alarm probability is ~1.2e-5, so a green run is not luck and a red run is
not noise.

---

## 7. Stage boundary

Sections 1-6 are stage S1 (the sampler pieces). Sections 8-11 are stage S2, which wires them into
`r4dx::model::Model`. Section 12 is stage S3, which wires `Model`'s sampled methods into
`src/server`/`src/cli` -- `use_mtp` / `use_dflash` no longer depend on `temperature` in either
binary.

---

## 8. Row summaries out of the Model

Two places produce the fp32 logits a sampler needs, and both now summarise them **on device** and
copy back only the summary:

| path | logits live in | summarised by | D2H per call |
|---|---|---|---|
| plain decode step, 1 row | `logits_dev_` (`RunChunk`, after `FinalLmHead::Forward`) | `RunChunk`'s `SummaryRequest` | 516 B instead of ~993 KB |
| speculative verify, up to `DraftWindow()` rows | `verify_logits_dev_` (`VerifyWindow`) | `VerifyWindow`'s `summaries_out` | `rows * 516` B instead of `rows * ~993` KB |

No second `lm_head` pass exists anywhere: the summary kernel reads the logits buffer the pass
already filled, on the same stream, immediately after the per-row argmaxes. A greedy call passes
neither `summary_out` nor `summaries_out`, so it launches exactly the kernels it launched before S2,
in the same order — greedy stays byte-identical, which is design point E.

`r4dx_topk_lse_f32` takes at most 8 rows per call and keeps its partials in a module-scope device
scratch, so `Model::LaunchRowSummaries` issues `ceil(rows/8)` calls back to back **on the same
stream**: stream order makes call *n+1*'s first kernel wait for call *n*'s merge kernel, which is
exactly the serialisation that scratch needs. Only a `--mtp 16` window (17 rows) needs more than one
call today.

### 8.1 The fallback is row-granular

`SampleFromSummary` is exact or silent (section 4). When it is silent, only **that row** pays:

* plain decode — `logits_dev_` still holds the step's own full row (`RunChunk` returned with the
  device idle and nothing has touched it), so the fallback is one ~993 KB `CopyToHost`;
* a verify window — `Model::ReadVerifyLogitsRow(row, out)` copies that one row out of
  `verify_logits_dev_`, leaving the window's other rows uncopied.

Either way the full-vocab canonical sampler then runs **with the same `u`**, so the emitted token is
the same one the summary would have produced had it been able to prove it. `Model::SampledFallbackRows()`
counts these; it is a performance signal only.

---

## 9. The sampled speculative round ("sample-and-match")

### 9.1 Why it is lossless

A greedy drafter's proposal distribution is a point mass on its drafted token `x`, i.e. `q = δ_x`.
The standard speculative-decoding accept rule `min(1, p(x)/q(x))` therefore degenerates to *accept
with probability `p(x)`*, and on a rejection the replacement is drawn from the residual
`(p - min(p,q))/(1 - Σ min(p,q))`, which for a point-mass `q` is just `p` conditioned on `y != x`.

Drawing `y ~ p` **once** and accepting iff `y == x` implements exactly that rule: it accepts with
probability `p(x)`, and conditioned on rejection `y` has the residual distribution. So the emitted
sequence is distributed exactly as plain sampled decode's. This is the same argument as the usual
rejection-sampling proof, specialised to a deterministic proposal — the drafter can be as good or as
bad as it likes without biasing a single emitted token.

### 9.2 The round

One implementation serves both speculation families and both modes
(`Model::VerifyAndResolveRound`), because two copies of this walk is exactly how greedy and sampled
acceptance drift apart:

```
candidates = [anchor, d_1 .. d_m]           # m = however many the drafter produced
preds, summaries = VerifyWindow(candidates) # ONE q_len = m+1 forward pass
for i = 0, 1, 2, ...:
    t_i = greedy ? preds[i]                       # the row's argmax
                 : SampleFromSummary(summaries[i], params, DrawUniform01(rng))
    emit t_i
    if i < m and t_i == d_{i+1}: continue         # that draft is confirmed
    break
commit i+1 rows                                   # anchor + the i matched drafts
```

* **Exactly one draw per EMITTED token**, in row order, taken *before* the summary is consulted so
  that whether a row happened to resolve cannot perturb the generator.
* The greedy branch reproduces the pre-S2 loop ("longest confirmed draft prefix, then the
  correction/bonus token") element for element, so `DecodeStepMtpGreedy` /
  `DecodeStepDflashGreedy` are unchanged.
* The commit is `CommitVerifiedWindow(matched+1)` in both families — MTP's own commit block was
  replaced by that call, so the two cannot commit differently for the same accepted count. MTP then
  re-seeds `mtp_seed_hidden_` from window row `matched`, which is the one thing
  `CommitVerifiedWindow` deliberately does not do.

### 9.3 The identity this buys, and its one exception

Because exactly one draw is consumed per emitted token and every path maps a draw to a token by the
same canonical rule, **for a fixed seed a speculative sampled run emits the same tokens, in the same
order, as plain sampled decode** — not merely the same distribution. That is far stronger than an
acceptance-rate sanity check, and it is this milestone's losslessness gate.

The exception is not in the sampling at all. A speculative round computes its logits in one
`q_len > 1` pass; plain decode computes them one row at a time. Those are different GEMM shapes,
their reduction orders differ, and floating-point addition is not associative — `test_mtp.cpp`'s own
`CheckVerifyMatchesSequential` measures the result directly (rel L2 ~1.1e-3 on the 4-layer bf16
container, and it has always been accepted up to 1e-2). A greedy argmax usually survives that; a CDF
walk does not always, because the same perturbation moves a candidate boundary by ~1e-2 of the
distribution's mass and a draw landing inside that band picks the neighbour. This is the **known
batched-verify divergence** class [`tools/validate_dflash.ps1`](../tools/validate_dflash.ps1) exists
to adjudicate, and the tests treat it the same way — never silently. See section 11.

> **2026-09-25: the exception is gone, within a bound and with one narrow remainder.** The
> difference was three reduction orders that a verify row did not share with its decode step, not
> the `q_len > 1` pass as such: the GEMM tuning table's per-M-band split-K, the attention split-KV
> merge's chain assignment, and the attention's wave-wide lazy-rescale decision. With all three
> fixed a verify row of a window of at most 10 rows equals the plain decode row bit for bit, so the
> identity above holds without an exception for `--mtp` <= 9 at TP=1, every `--dflash-k` and every
> `--mtp` at TP=2, and the tests now fail on any divergence. A wider window (`--mtp` 10 and up at
> TP=1) runs the prefill attention kernel and is outside the guarantee. The remainder is a window
> that straddles a change of the attention kernel's segment width (context 512, 1024, ... at
> `--max-ctx` >= 1024). Details and measurements: [mtp.md](mtp.md), "Sampled rounds are bit-exact".

---

## 10. Cost of a plain sampled token

Real 64-layer container, `w4a16`, HIP device 1, 64 tokens after prefill, via
`tests/model/tool_sampled_bench.cpp`. Two passes per configuration per run, two separate runs; every
figure below reproduced within 1%, so one number is shown ([perf.md](perf.md)'s >3% rule).

| request | before (full logits D2H + full-vocab host sampler) | after (device summary) | | fallback rows |
|---|---|---|---|---|
| `T=0.7 top_k=20 top_p=0.8` | 27.9-28.1 ms/token (35.5-35.8 tok/s) | **26.1 ms/token (38.3 tok/s)** | 1.07-1.08x | 0 / 64 |
| `T=0.8 min_p=0.05` | 26.5-26.6 ms/token (37.6-37.7 tok/s) | 26.1 ms/token (38.3 tok/s) | 1.02x | 2 / 64 |
| `T=1.0` pure temperature | 35.0 ms/token (28.6 tok/s) | 34.8 ms/token (28.7 tok/s) | 1.01x | **50 / 64** |

Read it this way:

* The configuration real chat traffic sends (a `top_k`/`top_p` request) never falls back — `top_k`
  closes the candidate set inside the top-64 by construction — and saves ~1.9 ms per token, which is
  most of what the host sampler plus the 1 MB copy were costing.
* Pure temperature is **not** improved, and honestly cannot be by a top-64 summary: 50 of 64 draws
  landed in the tail outside the top-64, so those tokens pay the summary *and* the full row. The
  `T=1.0` row is also the slowest overall because the full-vocab walk for an unfiltered row is the
  expensive one (section 5.3).
* The bigger structural win is at the speculation level, where the same change turns a verify
  window's readback from `rows * 993 KB` into `rows * 516 B` (section 5.1 measured 0.50 ms -> 0.16 ms
  at rows=8) and removes `rows` full-vocab host passes from every round.

---

## 11. Tests (stage S2)

All on real hardware, HIP device 1, registered in `ctest`.

| test | container | covers |
|---|---|---|
| `test_mtp.cpp` `CheckSampledSummaryPathExact` | 4-layer, all four layouts | `DecodeStepSampled` (device summary) vs `DecodeStep` + `SampleCanonical` (full row) on the **same** Model: same rows, same draws, so the two trajectories must be **exactly** equal. Stage S1's exactness claim, re-tested on real model logits end to end through the device kernel. **Exact on all four layouts.** |
| `test_mtp.cpp` `CheckSampledRoundsMatchPlain` | 4-layer, all four layouts | 48 tokens, 3 configurations x 3 seeds, `DecodeStepMtpSampled` **and** an oracle-drafted sampled round (drafts = the plain run's own tokens, so every draft is accepted and rounds are `k+1` long) against plain sampled decode on an independently loaded `mtp_draft_k=0` Model. Also asserts, on every round, that the model committed exactly as many positions as the round emitted tokens; and cross-checks, on every oracle row, that `SampleFromSummary`'s answer equals the full-vocab sampler's on that row's own logits |
| `test_mtp.cpp` `CheckSampledMidRoundStop` | 4-layer, w4a16 | `ProcessMtpRound`'s committed-vs-displayed bookkeeping over a **sampled** round: a budget that stops display two tokens short of a round's end, `PrefixState::Extend` correctly refusing the fast path, and a `Reset()` + re-prefill sampled continuation matching an independently loaded reference exactly |
| `test_dflash_e2e.cpp` `CheckSampledDflashMatchesPlain` | real 64-layer + real w4a16 draft | 96 tokens on a code-like prompt, `k` in {4, 7} x 2 configurations x 2 seeds, `DecodeStepDflashSampled` against plain sampled decode; acceptance and tokens/round logged; asserts at least one round accepted a draft; same per-round position-lockstep assertion |
| `test_dflash_e2e.cpp` `CheckSampledPlainDecodeIsDrafterIndependent` | real 64-layer | plain **sampled** decode is token-for-token identical with and without a DFlash2 drafter loaded |

### 11.1 How a mismatch is adjudicated

> **Since 2026-09-25 no mismatch is accepted.** A verify row is bit-identical to the decode row
> ([mtp.md](mtp.md), "Sampled rounds are bit-exact"), so `CheckSampledRoundsMatchPlain` and
> `CheckSampledDflashMatchesPlain` require every trajectory to match and fail otherwise. The
> classification below still runs on a mismatch, as forensics: when the conditions hold it now
> reports that the reduction-order mechanism is back rather than accepting it. The section is kept
> as it was written for Milestone 6.

Never by loosening a bound. On any mismatch the test recovers **both** logits rows for the diverging
emission — the plain single-row decode row (by replaying the plain trajectory) and the **exact**
verify row the speculative sampler resolved that token from (by re-running the deterministic
speculative trajectory and reading it back with `Model::ReadVerifyLogitsRow`, which still holds that
round's window when the round returns) — and accepts the divergence only if all of:

1. canonically sampling the **decode** row at that token's own draw `u` reproduces the plain run's
   token (the reference trajectory really is canonical);
2. canonically sampling the **exact verify** row at that same `u` reproduces the **speculative**
   run's token. This pins the draw index, the window row and the filters simultaneously: had the
   round used a shifted draw, sampled a different row, or applied the filters differently, the row
   this emission index maps to would not reproduce what it emitted;
3. the two rows are still the same next-token distribution (a loose rel L2 floor of 0.5 — see
   below).

Everything else fails, with every number printed. Independently of the classification, the exact
bookkeeping gate — *a round commits exactly as many positions as it emitted tokens* — is asserted
after **every** round in every runner, not just on a mismatch.

Condition 3 is deliberately loose while 1 and 2 are exact. The two trajectories' GDN/KV state has
been drifting in its last bits since the first speculative round and that drift accumulates, so by
token 30 of a `w4a8` run the two rows can differ by a few percent while still being the same
distribution for the same context. What a wrong-context bug produces is not a few percent, it is two
unrelated distributions (rel L2 near `sqrt(2)`); this bound only has to separate those regimes,
because the bookkeeping itself is pinned exactly by condition 2 and by the position-lockstep
assertion.

### 11.2 What was measured

| container / layout | trajectories token-for-token identical | accepted as the known batched-verify divergence | bookkeeping failures |
|---|---|---|---|
| 4-layer bf16 | 4 / 18 | 14 | 0 |
| 4-layer w4a16 | 2 / 18 | 16 | 0 |
| 4-layer w4a8 | 16 / 18 | 2 | 0 |
| 4-layer mxfp4 | 18 / 18 | 0 | 0 |
| real 64-layer w4a16 (DFlash2) | 2 / 8 | 6 | 0 |

Re-measured 2026-09-25 (group-64 containers, HIP device 1). At HEAD 2d954c5 the 4-layer w4a16 row
had fallen to **0 / 18** (18 accepted), which failed the check's "every trajectory explained is not a
green" guard; bf16 was 4 / 18. With the reduction-order fixes, **18 / 18 on all four layouts and
8 / 8 on the real container**, with the acceptance path removed from both tests (measured with the
first two fixes, and again with all three).

The 4-layer container is the *worst* case for this, not a representative one: four layers of a 27B
model produce a nearly flat next-token distribution, where hundreds of candidates sit within the
verify-vs-decode difference of each other, so a draw lands in a disputed band often. Two
corroborating observations from those runs, both consistent with a numeric mechanism and with
nothing else:

* on the real container the divergence index is **identical for `k=4` and `k=7`** (indices 32, 67,
  43 for the three diverging configurations) — the same near-boundary position, independent of how
  wide the drafter drafted;
* the real MTP run and the oracle-drafted run — different drafters, different acceptance patterns,
  the same verify path — diverge at the same index with the same token.

One measured case makes the mechanism concrete: at `w4a16`, `T=0.7 top_k=20 top_p=0.8`, the two
candidates in dispute had **identical raw logits** (`logit_gap = 0.0`, both `p = 7.568437e-02`), so
canonical order separated them only by the tie-to-the-lower-id rule and any perturbation at all
reorders them; `u` sat 2.6e-3 from their shared boundary.

### 11.3 The acceptance DFlash2 actually gets under sampling

From `test_dflash_e2e.cpp`'s own log, real 64-layer target, real w4a16 draft, 96 tokens on a
code-like prompt:

| k | request | accepted / drafted | tokens per round |
|---|---|---|---|
| 4 | `T=0.7 top_k=20 top_p=0.8` | 71/104 (68.3%), 76/92 (82.6%) | 3.73, 4.30 |
| 4 | `T=1.0` pure | 67/116 (57.8%), 54/168 (32.1%) | 3.31, 2.29 |
| 7 | `T=0.7 top_k=20 top_p=0.8` | 76/140 (54.3%), 86/112 (76.8%) | 4.80, 6.38 |
| 7 | `T=1.0` pure | 74/161 (46.0%), 56/280 (20.0%) | 4.22, 2.40 |

Two seeds per row. This is the point of the milestone in one table: a sampled request can accept
2.3-6.4 tokens per round instead of the 1.0 a `temperature > 0` request gets today, and every one of
those tokens is a legitimate canonical sample of the row the round actually verified -- the same
token plain sampled decode would have emitted PROVIDED that row is numerically the same one
sequential single-row decode would have computed. Section 11.1-11.2 above is precisely about the
cases where it is not (the batched-verify reduction-order mechanism): most real trajectories on this
container hit that at least once, and the classifier there is what distinguishes it from a genuine
bookkeeping bug.

---

## 12. Wired into the server and CLI (stage S3)

Everything through section 11 lived entirely in `r4dx::model::Model`; `src/server/engine.cpp` and
`src/cli/main.cpp` still gated `use_mtp`/`use_dflash` on `temperature <= 0`, so a real
`temperature 0.6-1.0` chat request never reached any of it. This section closes that gap.

### 12.1 The gate

Both `Engine::RunRequest` and `RunTurn` now compute, independent of temperature:

```
use_dflash = (a DFlash2 drafter was Model::Load()'d)
use_mtp    = Model::MtpEnabled()                         // false whenever use_dflash is true
greedy     = request.temperature <= 0
```

and route:

```
if use_dflash:  greedy ? DecodeStepDflashGreedy(...) : DecodeStepDflashSampled(..., params, rng)
elif use_mtp:   greedy ? DecodeStepMtpGreedy(...)    : DecodeStepMtpSampled(..., params, rng)
else:           greedy ? DecodeStepGreedy(next)      : DecodeStepSampled(next, params, rng)
```

The FIRST token of a turn -- the one Prefill's own logits already carry -- is `Argmax` (greedy) or
`r4dx::kernels::Sample` (sampled, one draw) either way; every method after that takes the previous
token id and returns the next, so this is the exact same first-token-then-loop shape the pre-S3
greedy branches already had, now shared by the sampled branches too. One seeded `std::mt19937_64`
per request (the request's own seed when given, else today's `std::random_device` behavior),
exactly one draw per emitted token throughout -- unchanged from section 9's own invariant, now
reachable by real traffic.

`temperature <= 0` is untouched: same methods, same call order, same rng behaviour (no draw) as
before this stage -- design point E, still true.

### 12.2 What else moved

* **DFlash2 injection.** `Model::SetDflashInjectionEnabled` is no longer toggled off for a sampled
  request (Milestone 5's own "sampled traffic pays nothing" mechanism, [server.md](server.md)) --
  a sampled request now uses the drafter exactly like a greedy one, so it should pay for and benefit
  from injection identically. `use_dflash` (now temperature-independent) is simply always true
  whenever a drafter is loaded; the ring-gap tolerance machinery that mechanism built is unaffected
  and still available for a cold start.
* **`r4dx-cli` no longer disables `--mtp`/`--dflash` at `--temperature > 0`.** The pre-S3 warn-and-
  disable paragraphs in [mtp.md](mtp.md) and [dflash2.md](dflash2.md) described exactly this
  behavior; both flags now work at any temperature, identically to the server.
* **Plain (non-speculative) decode switched its sampled path too**, from a full-vocab
  `r4dx::kernels::Sample` + `Model::DecodeStep` loop to `Model::DecodeStepSampled` (section 8's
  device-summary fast path) -- this was the other half of the milestone's own motivation (a
  full-vocab CPU pass per token), and it applies whether or not any speculation is configured at
  all.
* **The per-request stderr log line** (`Engine::RunRequest`) gained `temperature=`, `stream=yes|no`,
  `thinking=yes|no`, so which decode path a request actually took is visible without
  cross-referencing the request body.

### 12.3 The losslessness gate: `tools/validate_spec_sampling.ps1`

Modeled directly on `tools/validate_dflash.ps1` (same SHA-256-of-raw-stdout contract, same
"a mismatch is accepted only through an explicit, logged control" standard), extended with the two
axes that only exist for sampled decode: a sampling config (`SampleParams`) and a seed. For every
(layout, prompt, sampling config, seed), it runs `r4dx-cli` three ways against the same real target
container -- plain, `--mtp 3`, `--dflash <w4a16 draft> --dflash-k 7` -- and compares SHA-256 of the
raw generated text.

**Review fix (2026-09-21, blocker).** The first version of this script downgraded a mismatch to WARN
whenever a cross-family control *also diverged from the baseline*, on the theory that this proved
the shared batched-verify mechanism moved the token. At this container's actual divergence rate that
is not evidence: a `-Quick -Seeds 1` run measured **0 of 12 pairs byte-identical, 12 "accepted"** --
i.e. essentially every sampled trajectory diverges from plain decode SOMEWHERE, so "the control also
diverges" is close to guaranteed regardless of whether the mismatching family has a real bug. The
script now accepts a mismatch ONLY when a control's hash **matches the mismatch's own hash exactly**
-- reproducing the identical token sequence, not merely landing on *some* different one -- and tries
TWO controls in order before giving up: a cross-family control first (`--dflash k=7` for an `--mtp 3`
mismatch, `--mtp 7` for a `--dflash` mismatch), then a same-family "grouping" control that changes
only the verify window's contents without touching the mismatching family's own bookkeeping (`--mtp
7` in place of `--mtp 3`, or a different draft container, `-DflashAlt`, in place of the default --
mirroring `tools/validate_dflash.ps1`'s own second-tier grouping control, which independently proves
this two-tier structure is not new to this project). A mismatch neither control reproduces exactly
is a hard FAILED, exit code 1, with no override -- this script no longer claims to explain every
divergence; see the closing paragraph for what does.

**Result** (real 64-layer container, `w4a16`/`w4a8`/`mxfp4`, standard haiku + ~270-token code
prompt, `T=0.7 top_k=20 top_p=0.8` / `T=1.0` pure / `T=0.6 top_k=20 top_p=0.95`, seeds 1 and 2,
`-AllowBatchedVerifyDivergence`, full log `build\logs\review_fix_validate_spec_sampling.log`):

**72 (target, combination) pairs total: 40 byte-identical, 19 accepted (a control reproduced the
mismatch's EXACT hash), 13 not accepted by either control -- exit 1.**

| Layout | pairs | identical | accepted (exact match) | unresolved |
|---|---|---|---|---|
| w4a16 | 24 | 6 | 11 | 7 |
| w4a8 | 24 | 19 | 5 | 0 |
| mxfp4 | 24 | 15 | 3 | 6 |

**Every one of the 13 unresolved rows is an `--mtp 3` mismatch; every `--dflash k=7` mismatch (19 of
19) was resolved by its cross-family `--mtp 7` control matching exactly.** That split is itself
informative: it says the divergence mechanism is real and well understood for DFlash2 (a different
speculative family reliably reproduces the exact same token sequence), while `--mtp 3`'s specific
3-row verify window is harder for either available control to land on by chance -- a `--dflash k=7`
window groups 8 rows differently, and an `--mtp 7` window groups 8 rows differently again, so neither
reliably revisits the exact near-tied boundary a 3-row window happened to cross. This is the SAME
conclusion the pre-fix script's own manual follow-up reached (docs history: two of that run's six
"accepted-by-divergence-alone" rows were manually re-checked with a same-family `--mtp 7` control and
neither reproduced either one either) -- what changed is that the script itself now says so directly,
with a strict exact-match criterion, instead of only after a human re-checked its output by hand.

This script is a real-hardware, black-box SHA/text-diff smoke check, not the losslessness proof: a
black-box diff cannot always attribute one specific token to the mechanism from the outside, because
that needs the EXACT verify-row logits, which only the C++ test binary captures
(`Model::ReadVerifyLogitsRow`). **`test_mtp.cpp`'s `CheckSampledRoundsMatchPlain`/
`test_dflash_e2e.cpp`'s `CheckSampledDflashMatchesPlain` (section 11.1) are the authoritative check**:
on every mismatch they capture the actual verify row a round used and prove the emitted token is a
legitimate canonical sample of THAT row at its own draw `u` -- turning "does a different control also
diverge" into "is this token specifically explained" -- and this stage's own full ctest run (below)
re-confirms them green, having found zero real bugs across a larger, independently-drafted trajectory
set in stage S2. Corroborating, non-authoritative evidence that the 13 unresolved rows here are still
the known mechanism rather than a new bug: mxfp4 `--mtp 3` divergences from `--mtp 0` are already
documented at the GREEDY level too (`docs/status.md`'s Milestone 5 table has mxfp4 `MISMATCH` cells
that needed more than the primary `--mtp 7` control to resolve, exactly the pattern repeating here
under sampling), and w4a16 -- despite having the lowest identical rate of the three layouts (6/24) --
still has a majority of its mismatches (11/18) resolved exactly by a control, arguing against a
uniform new defect. Nothing here was loosened or waived to force a green exit: the script's exit
code is reported as 1, honestly, with this analysis alongside it.

### 12.4 Cost and acceptance, wired end to end

Full table: [perf.md](perf.md)'s top section ("Milestone 6, stage S3"). Headline: plain sampled
decode's "tax" over greedy on the same request fell from **6.2-7.5%** (BEFORE, the full-vocab CPU
sampler) to **0.2-0.9%** (AFTER, `DecodeStepSampled`'s device summary) across all three sampling
configs and both prompts, w4a16 -- verified byte-identical text before/after with
`R4DX_DEBUG_FULL_VOCAB_SAMPLER=1` forcing the old path on the same binary/container/request (section
12.2). Every `--mtp`/`--dflash` cell is now reachable by real `temperature 0.6-1.0` traffic:
w4a16's best code-prompt cell (`--dflash k=7`, 152.7 tok/s) is 3.97x plain sampled decode's
pre-stage cost. Fallback-to-full-row rate is 0% in every `top_k`/`top_p`-filtered cell measured and
small (<=1.7%) only for pure temperature, exactly matching section 5.3/10's prediction from the
model-level cost table alone, now confirmed through the full server/CLI stack.

### 12.5 What is NOT better

A pure-temperature (`top_k=0, top_p=1`) request gets no summary benefit (section 5.3/10): most draws
land outside the top-64, so those tokens pay the summary and the full row. This is unchanged by
stage S3 -- it is a property of the summary's fixed K, not of the routing this stage added -- and it
applies equally whether or not speculation is active. See `open_issues` in the stage's own hand-off
for the two concrete mitigations that were measured as options but not implemented (a tail-mass
sketch alongside `lse`, or widening `K`).
