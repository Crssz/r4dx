# Real-model reference of qwen4_exp (plan M0-5 full depth, M0-7)

Reference logprobs, probes, layer goldens and fp8 KV calibration of `Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ`, produced by
the layer-streaming reference (`ref.py`: HF's own decoder layers, AWQ int4 dequantized to bf16 on the GPU per layer, experts
lazily by routed set, n-gram rows from the real sidecar) on device 1. Everything below was run on 2026-10-10/11 and every
number is an observed one. Outputs live under `E:\models\r4dx\q4exp\ref\` (`MANIFEST.json` there has sizes + sha256 of every
file; the large ones are listed below).

## Gate evidence against the fp32 truth and a noise-relative gate proposal (read first)

`ref.py` has a `dtype=torch.float32` switch (default bf16 unchanged: `test_ref.py`, `test_awq_dequant.py`, `test_real_ref.py` 36 passed; the real
english_prose canon dump from the edited `ref.py` is byte-identical to before, sha256 `6a8ffd04...41`). It runs the same HF modules with fp32 weights
(int4 dequant `(q - zp) * scale` in fp32, UNROUNDED: `awq_dequant_f32.py`), fp32 activations/residual, fp32 QSA selection scores, router and logits
(no TF32: `set_float32_matmul_precision("highest")`). That is the "fp32 truth" of the Gemma lane (docs/gemma4-plan.md 9.9-9.11): the same network
with the bf16 arithmetic taken away; its dumps are `truth-fp32\` (recent rule; canon 4 segments + probe_8k), `truth-fp32-w16\` (same, but the
weights rounded to bf16 exactly as the bf16 stack and r4dx's converter see them, fp32 compute) and `truth-fp32-hf\` (probe_8k, HF-exact tie rule).
Everything below is `kl_report` arithmetic (fp64, full 248,320 vocabulary); `truth_gate.py` reproduces the tables.

**1. Stock-bf16 noise against the truth** = what a correct bf16 implementation of this model shows (KL(truth || bf16), nats; canon = 4 x 1023 rows):

| bf16 run vs `truth-fp32` | rows | mean KL | median | p99 | max | top-1 |
|---|---:|---:|---:|---:|---:|---:|
| (a) canon, the reference | 4092 | **0.01386** | 0.00418 | 0.1104 | 0.649 | **95.19%** |
| (b) canon_p1 (same stack, +1 token) | 4092 | 0.01428 | 0.00421 | 0.1197 | 1.609 | 95.28% |
| (a)+(b) pooled, the gate's noise term | 8184 | **0.01407** | 0.00419 | 0.1164 | 1.609 | 95.23% |
| (a) probe_8k, the reference | 8191 | **0.01570** | 0.00185 | 0.1501 | 6.607 | **96.34%** |
| (b) probe_8k_p1 (+1 token) | 8191 | 0.01561 | 0.00171 | 0.1437 | 7.340 | 96.37% |
| probe_8k, HF-exact tie rule (bf16) | 8191 | 0.01312 | 0.00184 | 0.1634 | 1.816 | 96.23% |

Per canon segment (reference run; +1 run): cpp 0.01338 / 95.50% (0.01199 / 95.5%), english 0.00974 / 95.5% (0.00905 / 95.8%), python 0.01663 / 95.8% (0.01970 / 95.7%),
thai 0.01567 / 93.94% (0.01639 / 94.13%). With `truth-fp32-w16` as the truth the numbers are the same within 0.001 (canon pooled 0.01474 / 95.23%, probe_8k
0.01367 / 96.25%; `noise_floor\truth_gate_w16.json`).

**2. Two bf16 runs against each other** (c): canon vs canon_p1 0.01092 / 95.80% (pooled 4092 rows), probe_8k vs probe_8k_p1 0.01107 / 96.72% (see the controls
table below). Less than the sum of two truth distances (0.028 on canon): the bf16 errors of two runs are largely shared (the same weights, the same rounding
points), but each run is ~0.014 from the truth.

**3. What is the floor made of** (each is a perturbation of ONE kind; the stack is chaotic enough that any bf16-sized perturbation saturates around 0.01):

| perturbation | KL (nats) / top-1 |
|---|---|
| bf16 arithmetic everywhere (bf16 reference vs truth), canon / probe_8k | 0.0139 / 95.2%, 0.0157 / 96.3% |
| the same bf16 stack with one more token (shape/batch-composition only) | 0.0109 / 95.8%, 0.0111 / 96.7% |
| AWQ weights rounded to bf16 (`truth-fp32` vs `truth-fp32-w16`, both fp32 compute), canon / probe_8k | 0.00866 / 96.53%, 0.00801 / 97.23% |
| **QSA tie rule recent vs HF-exact in fp32** (`truth-fp32` vs `truth-fp32-hf`), probe_8k | **0.00073** / 99.46% (median 4e-6, p99 0.0118, max 0.865, 0 rows > 1 nat) |
| QSA tie rule recent vs HF-exact in bf16, probe_8k / probe_32k | 0.01233 / 96.91%, 0.01154 / 97.06% (= the bf16 floor, not the rule) |

So the STOP flag below is resolved: the tie rule's own noise floor is 0 on canon and 0.0007 at 8k; what exceeded 0.01 is the bf16 floor, which is
a property of the model in bf16, not of the rule or of this tooling. (The weight rounding to bf16 alone costs 0.0087: the pipeline Q3b/Q11 starts there.)

**4. Expectation for a correct r4dx bf16 implementation** ([INFERENCE]: r4dx has no unquantized path to measure; the stock bf16 HF stack stands in for it
and r4dx's kernels will round at different places, which is exactly what the +1-token control emulates):
* against the bf16 reference (stock-bf16 pair): mean KL ~0.011 (canon segments 0.008-0.014, thai highest), top-1 ~95.8% (94.0-96.6%); 8k: ~0.011 / ~96.7%.
* against the fp32 truth: mean KL ~0.014 (canon segments 0.009-0.017), top-1 ~95.2% (93.9-95.8%); 8k: ~0.016 / ~96.3%. A fp32 residual stream (the Gemma
  lane's remedy) would only move it down, towards the weight-rounding floor 0.009.
* Consequence for the standing gate (plan Q10: mean KL <= 0.02 and top-1 >= 93% against the bf16 reference, canon + probes): it passes an UNQUANTIZED
  bf16 r4dx with 0.009 nats / 2.8 points of slack and leaves ~0.009 nats for the trellis error at 3.3 bits, on corpora where the noise term alone is 0.011.

**5. Proposed gate** (the Gemma rule, `kl_report.gate_thresholds`): the reference is `truth-fp32` (below); per group the thresholds follow the stock-bf16 noise term
`N` (mean KL and top-1 of KL(truth || bf16), pooled over runs (a) and (b)):

    KL(truth || r4dx)        <= 1.5 * N_KL + 0.005          (mean over the scored rows)
    top-1(truth, r4dx)       >= N_top1 - 1 point

| group (canon rows) | rows (a)+(b) | N_KL | N_top1 | gate: KL <= | gate: top-1 >= |
|---|---:|---:|---:|---:|---:|
| english (english_prose) | 2046 | 0.00940 | 95.65% | **0.01909** | **94.65%** |
| thai (thai_prose) | 2046 | 0.01603 | 94.04% | **0.02904** | **93.04%** |
| code (python_source + cpp_source) | 4092 | 0.01543 | 95.63% | **0.02814** | **94.63%** |
| ALL (the 4 segments) | 8184 | 0.01407 | 95.23% | **0.02610** | **94.23%** |
| probe_8k (report; the same rule) | 16382 | 0.01565 | 96.36% | 0.02848 | 95.36% |

Reading: against the truth the gate is 0.012 nats above the stock noise on canon ALL (about the Gemma lane's 0.01 increment budget), and the top-1 floor
is the stock noise minus one point, instead of a 93% floor that a noise-only run sits 2.2 points above (canon ALL). `truth-fp32-w16` gives ALL 0.02710 / 94.23%.
Not covered: probe_32k has no truth dump (not run), so the 32k probe stays against the bf16 reference with the noise term of section 2 (report only). Using the gate needs no rerun of the references:

    D:\venvs\r4dx-qwen4-ref\Scripts\python.exe tools\reference\kl_report.py --ref-dir E:\models\r4dx\q4exp\ref\truth-fp32 `
        --test-dir <r4dx dump dir> --tokens tools\reference\kl_corpus\tokens_canon.json --out kl_truth.json          # compare to the table
    (probe_8k: --ref-dir ...\truth-fp32 --tokens ...\probes\tokens_probes.json --segment probe_8k)

Truth dumps (`MANIFEST.json` has every file and the metas). `truth-fp32\`: cpp_source.logprobs.f16 508,062,720 bytes `0b504742c3c9ec796fed903f2c45bc670c79a15f25a2cc0e4336d8ab897cf555`,
english_prose 508,062,720 `a2038dac5204b0d80dbd7467bbc1c0e47ee26a31428584c5c0ca270592ffea6c`, python_source 508,062,720 `df0c34f3f8c504858316c907ddb02c120f228ab138ea49aaaf4da6a7559a939d`,
thai_prose 508,062,720 `40c0ad8d1b9428a1b0cb370ea5502a377befdbb5a475b3a2a62d80e2f3f3a540`, probe_8k 4,067,978,240 `4fc36286917dff2f59db8ad4936ec96d5f2c7e00aff06bfbcf5b6d50dfce87d6`;
`truth-fp32-w16\probe_8k` `39173f16f873fca4b20201feaccbe68c7bab80aaad639079cbda44228f059b43` (canon: cpp `617d45f3...`, english `99c2dd7d...`, python `2c0ce44e...`, thai `8c781093...`);
`truth-fp32-hf\probe_8k` 4,067,978,240 `13ccd06cb36fbf05566014096afabae7af834cdd21ff3b4cf38d5dd1baa4b22f`. Run cost (one step each, device 1): truth canon 206.0 s (4 sequences, peak VRAM
10.43 GiB, RAM 2.81 GiB), truth probe_8k 207.0 s (4.73 GiB, 3.11 GiB); w16 209.1 s / 206.5 s; hf-rule probe_8k 206.4 s (bf16 8k run: 180.5 s; the weight stream dominates)

## Earlier finding that led to the truth runs (STOP flag, resolved above)

The QSA tie-rule noise floor is **exactly 0 on the canon corpus** (4 segments x 1023 rows, `canon` vs `ref-hf`: mean KL 0.00000,
top-1 100.00%, the logprob files are byte-identical): a 1024-token sequence is dense (<= 2051 tokens), so the tie rule never
applies. At the long probes it is **above the 0.01 STOP threshold**:

| comparison (rows) | mean KL | median | p99 | max | top-1 | KL>1 |
|---|---:|---:|---:|---:|---:|---:|
| hf-rule vs recent-rule, probe_8k (8191) | **0.01233** | 0.00132 | 0.10595 | 7.946 | 96.91% | 8 |
| hf-rule vs recent-rule, probe_32k (32767) | **0.01154** | 0.00158 | 0.17934 | 5.781 | 97.06% | 10 |

BUT this is not tie-rule noise, it is the bf16 reference's own numerical noise. Controls (same rule, selections identical by
construction, only the sequence length differs by ONE appended token, so every GEMM / MoE-batch / chunk shape changes):

| control | rows | mean KL | median | p99 | max | top-1 |
|---|---:|---:|---:|---:|---:|---:|
| `recent` probe_8k vs the same run with 8193 tokens (first 8191 rows) | 8191 | **0.01107** | 0.00152 | 0.11646 | 6.441 | 96.72% |
| canon (1024) vs canon + 1 true next token, 4 segments pooled | 4092 | **0.01092** | 0.00315 | 0.10543 | 0.710 | 95.80% |
| `recent` probe_8k vs the first 8191 rows of probe_32k | 8191 | 0.01441 | 0.00159 | 0.14001 | 7.316 | 96.47% |
| `hf` probe_8k vs the first 8191 rows of probe_32k | 8191 | 0.01334 | 0.00174 | 0.12486 | 7.656 | 96.41% |

Per canon segment (KL, top-1): cpp_source 0.00816 / 96.38%, english_prose 0.00763 / 96.58%, python_source 0.01380 / 96.19%,
thai_prose 0.01408 / 94.04%. The same run twice is bit-identical (probe_8k recent: both runs sha256 `cafe938e...c4`), and the
layer-major engine equals the plain `full_logits_qwen4exp.py` bit for bit on the real model (english_prose, sha256 `6a8ffd04...41`),
so this is bf16 chaos in the model (4-stream residual, top-10-of-512 routing flips, QSA), not an engine bug.

Consequence for the gate (plan Q10/Q13/Q22): *any* bf16 implementation that is not bit-identical to this reference will show
mean KL ~0.011 and top-1 ~96% against it on the canon corpus before one bit is quantized; the gate "mean KL <= 0.02, top-1 >= 93%"
leaves ~0.009 nats / 3 points for r4dx's quantization on the canon corpus, and at 8k/32k the tie rule is not separable from this
noise (hf-vs-recent 0.0123/0.0115 is the same size as the shape control 0.0111). The gate needs revisiting with the user; the
Gemma precedent (an fp32 truth plus the noise of stock bf16 as the allowance) is the natural shape: see the section above for the fp32 truth runs and the proposal.

## Tools (this directory)

| file | what |
|---|---|
| `real_ref.py` | the engine: `prepare` (CPU), `step` (one bounded GPU step), `manifest`, `ngram-check`. Layer-major: layer i is built and its experts dequantized once, EVERY sequence of the jobset runs through it, residuals are checkpointed, then layer i+1 |
| `real_ref_run.ps1` | per-jobset driver; EVERY step is its own `tools\with_lock.ps1 -Name gpu` acquisition (device 1); ends a step at a layer boundary after `-MaxMinutes` (11) |
| `noise_floor.py` | `KL(hf \|\| recent)` + top-1 per segment through `kl_report.compare_segment/overall/markdown`; exit 2 and a banner above mean KL 0.01; subcommands `prefix` and `shape-noise` for the controls |
| `test_real_ref.py` | CPU tests on the tiny model (13 tests, 86 s; incl. the fp32 switch and the fp32 dequant of a real expert linear) |
| `truth_gate.py` | the noise-relative gate numbers against `truth-fp32` (stock-bf16 noise by group, gate thresholds) |
| `awq_dequant_f32.py` | fp32 (unrounded) int4 dequant for `ref.py dtype=float32` |

Commands (all from the repo root; the lock is taken inside):

```
powershell -NoProfile -ExecutionPolicy Bypass -File tools\reference\qwen4exp\real_ref_run.ps1 -JobSet all        # goldens canon canon_p1 probe8k probe8k_p1 probe32k kvcalib
powershell -NoProfile -ExecutionPolicy Bypass -File tools\reference\qwen4exp\real_ref_run.ps1 -JobSet probe32k   # one jobset
D:\venvs\r4dx-qwen4-ref\Scripts\python.exe tools\reference\qwen4exp\noise_floor.py `
    --hf-dir E:\models\r4dx\q4exp\ref\ref-hf --recent-dir E:\models\r4dx\q4exp\ref\canon --out E:\models\r4dx\q4exp\ref\noise_floor\canon
D:\venvs\r4dx-qwen4-ref\Scripts\python.exe tools\reference\kl_report.py --ref-dir E:\models\r4dx\q4exp\ref\canon `
    --test-dir <r4dx dump> --tokens tools\reference\kl_corpus\tokens_canon.json --out kl.json                 # the gate
D:\venvs\r4dx-qwen4-ref\Scripts\python.exe tools\reference\kl_report.py --ref-dir E:\models\r4dx\q4exp\ref\probes\recent `
    --test-dir <r4dx dump> --tokens E:\models\r4dx\q4exp\ref\probes\tokens_probes.json --segment probe_8k       # probes
D:\venvs\r4dx-qwen4-ref\Scripts\python.exe -m pytest tools\reference\qwen4exp\test_real_ref.py -q
```

### Resumability

State per job in `<root>\work\<jobset>\<job>\state.json` (ids sha256, tie rule, `layers_done`, residual file + its sha256) and
`resid_L<k>.bf16` (raw bf16 residual [1, T, 10240] after layer k-1, written atomically, previous one deleted after the state
commit). A restart verifies the residual's sha256 and the ids before continuing; a corrupt file or changed ids is refused.
Checkpoints every 4 layers and at the end of a step. Observed on the real model: the kvcalib jobset (37 sequences, 50,011 tokens)
stopped by its time budget at layer 47 after 623.5 s and resumed from the checkpoint in a second step (19.5 s). The same path on
the tiny model, one layer per step, is bit-identical to the single-step run, to `Qwen4ExpStreamingReference.logprobs_to_file`
and to `run_layers` truncated at the checkpoint (`test_real_ref.py`).
`reference_run.json` / `tokens_used.json` / `<segment>.meta.json` of every output directory follow `full_logits_qwen4exp.py`
(`kl_report.py` format: row i = log p(next | tokens[0..i]), fp32 log_softmax floored at -1e4, fp16, `[T-1, 248320]`).

### Efficiency work (all in my files; `ref.py` untouched)

* layer-major jobsets: one weight stream per layer for all sequences (a per-layer cache of dequantized linears, <= 10 GiB, cleared
  per layer). Plain path: 85.5 s stack + 2.4 s lm_head for ONE 1024-token segment (1.8 s/layer); 8 canon sequences (4 segments x
  2 rules) layer-major: 256.8 s in total. A 4096-token segment is not part of the corpus (the canon segments are 1024 tokens).
* the tie rule is read from `ref.tie_rule` at call time, so one built layer serves jobs of both rules.
* n-gram rows: `UnbufferedSidecarRows` (Windows `FILE_FLAG_NO_BUFFERING | RANDOM_ACCESS | OVERLAPPED` sector-aligned reads, adjacent
  rows merged, 16 threads) instead of the `np.memmap` of the 95 GiB file; same bytes as the memmap (`ngram-check`: 32,000 random
  rows, equal; 3.76 s vs 5.58 s cold). The 32k probe read 3.08 GiB in 695,636 reads; process peak working set stayed <= 3.75 GiB
  (nothing mapped, nothing enters the file cache).
* the GDN fallback of HF (no `flash-linear-attention` / `causal_conv1d` in the venv) is what the numbers below include.

## Timing and memory (device 1, one step each, lock free at start; peak = max over the step)

| jobset | sequences | tokens | wall | peak VRAM | peak RAM (working set) |
|---|---:|---:|---:|---:|---:|
| goldens (4 x 70, all 48 layers) | 4 | 280 | 96.8 s | 2.67 GiB | 3.09 GiB |
| canon (4 segments x {recent, hf}) | 8 | 8,192 | 256.8 s | 5.60 GiB | 3.00 GiB |
| canon_p1 (control) | 4 | 4,100 | 222.4 s | 5.50 GiB | 3.00 GiB |
| probe8k ({recent, hf}) | 2 | 16,384 | 229.4 s | 8.54 GiB | 3.18 GiB |
| probe8k_p1 (control, recent) | 1 | 8,193 | 180.5 s | 3.53 GiB | 3.17 GiB |
| probe32k ({recent, hf}) | 2 | 65,536 | 432.8 s | 18.26 GiB | 3.75 GiB |
| kvcalib | 37 | 50,011 | 623.5 + 19.5 s (2 steps) | 7.09 GiB | 2.17 GiB |

Per layer: ~3.3 s for one 8k sequence, 6.8 s for two 32k sequences (GDN) / 5.6 s (QSA),
logits + write: 35 s for one 32k sequence (16.3 GB), 9.8 s for 8k. Every step is far under the 15 minute rule, so the 32k probe
needed one lock acquisition (for both rules). Extrapolation: 8 canon-sized sequences cost 256.8 s layer-major against 87.9 s for one through the plain path, i.e. ~24 s
per additional sequence; the whole deliverable set above is ~34 minutes of GPU time (sum of the steps; the wall time of a run
also contains the wait for the lock, up to ~50 minutes when other agents held it).

## Deliverables (sizes in bytes, sha256)

Canon corpus (`tokens_canon.json`, 4 x 1024 tokens, tokenized with the canonical Qwen3.5 pre-tokenizer; 1023 rows each; 48 layers;
n_layers/meta recorded; `recent` rule in `canon\`, HF-exact rule in `ref-hf\`; both dense so byte-identical):

| file | bytes | sha256 |
|---|---:|---|
| `canon\cpp_source.logprobs.f16` (= `ref-hf\`) | 508,062,720 | b43a92ed7e65882f938ca5e3409f1d6fb65a82351894dc1ae218ed6385eea15e |
| `canon\english_prose.logprobs.f16` (= `ref-hf\`) | 508,062,720 | 6a8ffd04567b8bac9bd6cb1419b0b9b7d65ef51da4106383a2854e6593698441 |
| `canon\python_source.logprobs.f16` (= `ref-hf\`) | 508,062,720 | f43d5a40ca0041426941a570be752c35e61200957df5518a1959be0435210c9a |
| `canon\thai_prose.logprobs.f16` (= `ref-hf\`) | 508,062,720 | de95e4aa2057f440ce2a41116b0e92e2f0fa32e0509796838572cee92d57a30a |

Perplexities (recent): cpp 4.458, english 4.139, python 5.574, thai 6.128.

Probes (`probes\tokens_probes.json`, sha256 14cff94d8703ed5ec79fd7986935b46faa7b9e23672407e8aee88084eb891eab). probe_32k = the four
canon segments (cpp, english, python, thai = 4096 ids) followed by corpus-v2 (`E:\models\r4dx\corpus-v2\samples.jsonl`, rejected
rows dropped) documents round robin english_prose, code, chat (chat template, `add_generation_prompt=False`), thai_prose,
multilingual, each document NFC-normalized and encoded on its own (canonical tokenizer of
`E:\models\Huihui-Qwen3.8-27B-abliterated`), cut at 32768 ids (17 documents: `corpus_documents` in the tokens file). probe_8k = its first
8192 ids. `probe_8k_p1` = its first 8193 ids (control, `probes\tokens_probes_p1.json`). 8192 > 4096, so QSA layers use `ref.py`'s
query-blocked path (4096-query blocks, sdpa) and the vectorized indexer; selection at those lengths is non-trivial.

| file | bytes | sha256 |
|---|---:|---|
| `probes\recent\probe_8k.logprobs.f16` | 4,067,978,240 | cafe938e02ec65f8732399ac87071c8e22bb38da58cdf44ae3cb2dc65df937c4 |
| `probes\hf\probe_8k.logprobs.f16` | 4,067,978,240 | 011ca390722222070a9df3486ea131f22df8949b5fd978fb11d4360f4213a6f2 |
| `probes\recent\probe_8k_p1.logprobs.f16` | 4,068,474,880 | 63230dddcd8e95d945a11e3793b7456163d5fff65b5c87d3a672eedaa0699103 |
| `probes\recent\probe_32k.logprobs.f16` | 16,273,402,880 | d63068667f2d5ec48002850ccac698cb9b3a1531a7e312398518e3cc050a5fb0 |
| `probes\hf\probe_32k.logprobs.f16` | 16,273,402,880 | 36a168e6a7c80bf393d0f30fd5305706251c2306bbdf4f5352371a36998b0a61 |
| `canon_p1\{cpp_source,english_prose,python_source,thai_prose}_p1.logprobs.f16` | 508,559,360 each | see `MANIFEST.json` |

Perplexity probe_8k 3.115 (recent) / 3.116 (hf); probe_32k 2.722 (recent) / 2.742 (hf).

Layer goldens (`layer_goldens\`): `T70` = the first 70 ids of canon english_prose, `T70eos` = the same with EOS 248044 at positions 17
and 41; per tie rule `T70.{hf,recent}.safetensors`, `T70eos.{hf,recent}.safetensors` (hf == recent bit for bit: 70 tokens are dense).
161 tensors each, named exactly like the tiny goldens (`tiny_golden.py` docstring): `input_ids`, `embedding`,
`layer{i}.*` for i in 0..7 and 47 (`residual_in/out`, `attn_mix/inject/out`, `residual_attn`, `mlp_mix/inject`, `router_logits`,
`topk_ids/weights`, `shared_out`, `shared_gate_logit`, `moe_out`; layer 1: `ple_embedding`, `ple_ngram_ids`, `ple_out`, `residual_ple`;
QSA layers 3, 7, 47: `index_q/raw_keys/block_keys/mask`, `attn_q/k/v/gate/pre_o`), `mixer_out`, `logits` [70, 248320] bf16 (HF
bf16 lm_head). All finite; next-token ppl 9.50 (T70), 22.24 (T70eos). The tiny-model version of the capture reproduces the tiny
goldens tensor for tensor (`test_golden_jobs_reproduce_tiny_goldens`).

| file | bytes | sha256 |
|---|---:|---|
| `layer_goldens\T70.recent.safetensors` | 102,753,944 | 40e62bb65067faa228f553676cc90ee08a2d3b766e878348913cd9afadc99efd |
| `layer_goldens\T70.hf.safetensors` | 102,753,944 | 44467263ad26e140ddb6e70830dfac2ffe296fc37ee554b68331f391a615cebc |
| `layer_goldens\T70eos.recent.safetensors` | 102,753,944 | 8b926fcbe240b63dacaaebfbd9309f53a4fe442e357b4d2c8f1fe9b768359243 |
| `layer_goldens\T70eos.hf.safetensors` | 102,753,944 | 92f251e1e6bb885bcdcdb00547421087180755a40b92ce1a908144e072b2fc2b |

fp8 KV calibration `kv_calib.json` (118,480 bytes, sha256 c013a2f60891c95714389da1722a716c8af9b3a09c4a59e52dc5b752bce0ebaa):
`{"<layer>": {"k_amax": [2 floats], "v_amax": [2 floats], ...provenance}}` for layers 3,7,...,47, `ResolveKvDescale`'s format
(`descale = amax / 448`). Method = `kv_calibrate_full.py` ported to the streaming reference: 37 documents (the `kv_calibrate_full`
corpus: `calib.txt`, `kv_calib_corpus\*` incl. the chat conversation, + corpus-v2 documents round robin english_prose / code / chat /
thai_prose / multilingual), each its own sequence from position 0 through all 48 layers, truncated to 2048 tokens, 50,011 tokens in
total; K after `k_norm` and rope (what the paged cache stores), V at the `v_proj` output; `k_p9999`/`v_p9999` (exact 99.99th
percentile) and the corpus listing are in the file. Observed amax (k, v per kv head): layer 3 k 7.0/6.47, v 5.53/5.13; layer 27 k
8.69/9.69, v 33.0/21.0; layer 35 v 23.5/30.1; layer 47 k 15.9/10.9, v 8.13/7.38; the full table is in the file. Caveat: documents are
<= 2048 tokens, long-context amax (8k+) may be larger than what a static descale from this corpus saturates at; ratio amax/p99.99
reaches 13 on V (layer 27 head 0: 33.0 / 2.47).

Noise-floor outputs: `noise_floor\canon.{json,md}`, `probe_8k.{json,md}`, `probe_32k.{json,md}`, `shape_noise_8k_vs_8k_plus1.json`,
`shape_noise_canon_vs_canon_p1.json`, `prefix_8k_in_32k_{recent,hf}.json`.

## Verified vs not

Verified (run): CPU tests (13 in `test_real_ref.py`, plus `test_ref.py` and `test_awq_dequant.py`: 36 passed; `kl_report.py --self-test` unchanged); the
real-model layer-major run equals the plain streaming reference bit for bit (canon english_prose); determinism of repeated GPU runs;
`kl_report.compare_segment` self-comparison KL 0 on the tiny dumps; real dumps read by `kl_report.py` (noise-floor runs above).
Not done: an fp32 32k truth; HF's own `Qwen4ExpForConditionalGeneration` on the real weights (it does not fit; equality with
HF is the tiny-model test of `test_ref.py`); the `hf` rule on the GPU reproduces *torch.topk on device 1*, not on the CPU.

## Needs from others

Nothing blocking. (`ref.py` got the additive `dtype` / `fp32_exact_weights` switch; default behaviour is bit-identical.) Optional patches that would simplify this (the workarounds are in `real_ref.py`): `ref.py` `_build_layer` could read
`self.tie_rule` at call time instead of closing over it (the QSA `ix.forward` lambda), and `run_layers` could take a start layer + start
residual (the checkpoint path), and `CheckpointReader.linear_weight` could cache per layer.
