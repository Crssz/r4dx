# r4dx validation ladder

Five rungs, each one only trustworthy once the rung below it passes. A failure at rung N should
almost always be root-caused by re-checking rung N-1's tolerances, not by loosening rung N's.

```
1. packer byte-exact        src/convert's own round-trip test
2. kernel vs CPU/torch       tests/ (C++ side) vs a plain fp32 CPU reference, per r4d kernel
3. layer vs transformers     tools/reference/layer_golden.py's goldens vs src/model's layer graph
4. full logits (offloaded)   a handful of real tokens, whole model, HF transformers vs r4dx
5. generation sanity         does it actually produce coherent text at a real context length
```

## Rung 1 -- packer byte-exact

`src/convert` (HF safetensors + `config.json` -> r4dx container, `docs/container-format.md`) round
-trips: convert a tensor, read it back through the same permutation/dequant math the loader will
use, and compare against the pre-permutation source tensor bit-for-bit (for `bf16`/fp32 passthrough
tensors) or within a fixed ULP bound (for the quantized layouts, where round-to-nearest is
inherently lossy but *deterministic* -- the test is "does the packer produce the same bytes twice",
not "is quantization accurate", which is rung 3's job). Owned by `src/convert`'s own test suite,
not this directory.

## Rung 2 -- kernel vs CPU/torch

Each of the 15 linked `libr4d` units, in isolation, against a plain fp32 CPU reference
implementation of the same math (`tests/smoke_r4d.cpp`'s `r4d_gemm_bf16_nt_m64` check --
`rel_err < 2e-2` against a CPU fp32 matmul -- is the first instance of this rung; every other
kernel needs its own). This is pure C++ (or the `r4d.pyd` Python extension cross-check
`C:\Users\user\dev\libr4d\build-win\r4d.pyd` mentioned in libr4d's own docs) -- nothing in
`tools/reference` participates here directly, but the *tolerances* this rung should use for the
GDN/attention kernels come from comparing `libr4d`'s kernels against the intermediates
`layer_golden.py` captures (rung 3's job overlaps here for kernels that don't have a trivial CPU
reference, e.g. `r4d_gdn_chunk_scan_k128_v128_c64_bf16`'s fused WY-recompute-plus-recurrence --
"CPU reference" for that one *is* `torch_chunk_gated_delta_rule`, i.e. rung 3's captured
`*_gdn_q`/`*_gdn_k`/`*_gdn_v`/`*_gdn_g`/`*_gdn_beta`/`*_gdn_recurrent_state` tensors, fed through
the same math on CPU).

## Rung 3 -- layer vs transformers golden (this directory)

`tools/reference/layer_golden.py` builds one decoder layer at a time straight from `transformers`'
`modeling_qwen3_5.py` (real weights when the shard is downloaded, else seeded random init -- see
`tools/reference/README.md`), runs a fixed-seed `[T=64, hidden=5120]` prefill + `[T=4]` cached
decode, and dumps every intermediate `src/model`'s layer-graph tests need to diff against. Covers
layer 0 (GDN), layer 3 (full attention), `text.final_norm` + a tiny `lm_head` vocab slice, and the
MTP block's inner decoder layer.

For the **GDN layer**, both the RAW tensors the gated-delta-rule callee actually receives
(`gdn_q`/`gdn_k`/`gdn_v`/`gdn_g`/`gdn_beta` -- pre-l2norm, pre-scale, post-repeat_interleave to
`H=48` heads, `gdn_g` per-token) and the r4d-shaped tensors the linked kernels actually consume/
produce (`gdn_q_l2`/`gdn_k_l2`: l2-normed, `[T, Hg=16, K=128]`, NOT scaled by `1/sqrt(K)` --
`r4d_gdn_chunk_scan_k128_v128_c64_bf16` takes that as its own `scale` argument; `gdn_g_chunk_cumsum`:
`[T, H=48]` fp32, cumulatively summed within each 64-token chunk; `gdn_core_attn_out`: `[T, H=48,
V=128]`, the chunked-scan/recurrent-rule's raw output before the gated RMSNorm), plus the
causal-conv1d output in both untrimmed (`gdn_conv_out_fn_full`) and trimmed-to-`stage_len`
(`gdn_conv_out_fn_trimmed` -- the correct golden shape) forms, the gate/raw-a/raw-b projections
(`gdn_z`/`gdn_a_raw`/`gdn_b_raw`), and the gated-RMSNorm + out_proj outputs
(`gdn_gated_norm_out`/`gdn_out_proj_out`).

For the **full-attention layer** (and MTP's inner decoder layer, which is `full_attention`-typed):
the fused `attn.qg` raw projection, pre-rope normed q/k (`attn_q_normed`/`attn_k_normed`),
**post-rope** q/k (`attn_q_post_rope`/`attn_k_post_rope` -- captured via the same
`apply_rotary_pos_emb` monkeypatch `kv_calibrate.py` uses; this is what the fp8 paged KV cache
stores for K), the sigmoid gate (`attn_gate_sigmoid`) and pre-`o_proj` post-gate tensor
(`attn_post_gate`, isolating the `attn_output_gate` path), and the attention output.

Both layers' post-norm activations and MLP output are captured too (`post_input_norm`,
`post_attn_norm`, `mlp_output`, `layer_output`).

`src/model`'s test for a given r4d kernel call loads the matching `.safetensors` file +
`manifest.json`, feeds the golden's *input* tensor through the r4dx C++ layer graph on HIP device
1, and diffs against the golden's *output*/intermediate tensor using the tolerance table
`manifest.json["tolerances"]` carries:

| Tolerance | Applies to | Value |
|---|---|---|
| `bf16_matmul_rel_err` | anything ending in an `r4d_gemm_*` call (attn qkv/o, GDN in/out proj, MLP, lm_head) | `2e-2` |
| `fp32_elementwise_rel_err` | norms/gates/activations the reference computes in fp32 | `1e-4` |
| `recurrent_state_rel_err` | the GDN recurrent state specifically | `5e-2` (looser: accumulates chunk-sized drift) |

These mirror `tests/smoke_r4d.cpp`'s existing `rel_err < 2e-2` bf16-GEMM bound; don't invent a new
number per kernel without a reason recorded here.

`tools/reference/vision_golden.py` is this rung's vision-tower sibling (2026-09-20, `docs/vision.md`):
same conventions (real weights, `manifest.json`, the same tolerance table), covering patch embed,
all 27 encoder blocks (plus block 0's full internal kernel-granularity chain), and the merger, for a
real (synthetic but structured) test image processed through the checkpoint's actual configured
image processor. `tests/vision/test_vision_tower.cpp` is what diffs against it (2026-09-21).

`tools/reference/mrope_layer_golden.py` is this rung's sibling for the TEXT side of the vision
path (2026-09-22, `docs/vision.md` "Splicing pass: what was measured"): one real full-attention
decoder layer driven by the 3-axis `(t,h,w)` mrope position ids `Qwen3_5Model.get_rope_index`
produces for a prompt containing an image, at the same 2e-2 bound as `layer_golden.py`'s own
attention case and for the same reason. It exists because `layer_golden.py`'s position ids
collapse all three streams to one sequential index, so that golden passes identically whether the
rope kernel selects a per-bin position stream or ignores the h/w rows entirely. Consumed by
`tests/model/attention/test_mrope_attn_layer.cpp`, which also runs a NEGATIVE control (the same
call roping at the KV slot index) and asserts it lands well outside the band -- a tolerance test
is only evidence if the wrong answer actually fails it.

`tools/reference/kv_calibrate.py` is the same rung's sibling for the fp8 KV descale tables:
per-kv-head `amax` of K (post-rope) and V for a full-attention layer, which the converter turns
into `text.layers.{i}.attn.k_descale`/`.v_descale` (`amax / 448.0`, fp8 e4m3 max). It's explicitly
a **prototype** (see its `README.md` section and its output JSON's own `"caveat"` field) -- it
calibrates against raw token embeddings fed straight into one layer, not the true mid-stack
activation distribution that layer actually sees. Good enough to pin down the JSON contract and the
descale formula; the real per-model calibration pass (full 64-layer stack, real prompts) is
`src/convert`'s job, not this directory's.

### Known gaps at this rung (tracked, not yet closed)

- **`r4d_gdn_recurrent_update_k128_v128_bf16_fp32state`'s single-token decode path is not
  exercised.** `layer_golden.py`'s GDN decode step is `T=4` (per the task spec, matching
  `docs/architecture.md`'s speculative-window note), which takes the *chunked*
  gated-delta-rule path with a carried `initial_state`, not the dedicated single-token recurrent
  kernel's path (`use_precomputed_states and seq_len == 1` in `Qwen3_5GatedDeltaNet.forward`). A
  `T=1` decode variant is a cheap follow-up (`layer_golden.py`'s `run_text_layer` already threads a
  `decode_len` parameter -- running it a second time with `decode_len=1` and a fresh cache covers
  the gap) but wasn't added here to stay inside the task's scope.
- **The MTP block's fc/embedding-fusion path is not exercised**, only dumped raw
  (`mtp.fc.weight`, `mtp.norm.weight`, `mtp.pre_fc_norm_{embedding,hidden}.weight`) --
  `transformers` 5.17.0 has no Qwen3_5 MTP forward implementation to copy the math from
  (`Qwen3_5PreTrainedModel._keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`). The inner decoder
  layer itself (`mtp.layers.0.*`) *is* exercised with real weights. Whoever wires up
  `src/model`'s MTP path needs to derive the fc-fusion math from another source (the Eagle/DeepSeek-
  MTP papers this checkpoint's tensor names strongly resemble, or a newer `transformers` release
  that implements it) and extend `layer_golden.py` to match before that path can be validated here.
- **`kv_calibrate.py`'s calibration distribution is not representative** (see above) -- treat its
  `k_amax`/`v_amax` as a contract/format check, not a value to ship in a real container's
  `k_descale`/`v_descale` without re-running a proper full-stack calibration. Its output JSON
  records `torch_dtype`/`device` (amax differs materially between bf16-on-GPU and fp32-on-CPU) and
  merges into any existing `--out` file rather than overwriting it, so calibrating all 16
  full-attention layers is a matter of re-running it once per layer against the same output path.

### Fixed since the initial Opus review (kept here for traceability, not as open gaps)

- The GDN captures were previously mislabeled as "post-l2norm/repeat-interleave" when they were
  actually pre-l2norm, pre-scale, post-repeat_interleave (H=48) with a per-token (non-cumsum) `g`
  -- and the chunked-scan output, the gate, the raw a/b projections, and the gated-RMSNorm output
  were never captured at all. `layer_golden.py` now records both the raw form and the r4d-shaped
  form (l2-normed `[T,Hg,K]` q/k, chunk-cumsum'd `[T,H]` g, and the `[T,H,V]` chunk-scan output),
  plus `gdn_z`/`gdn_a_raw`/`gdn_b_raw`/`gdn_gated_norm_out`/`gdn_out_proj_out`.
- The decode-stage GDN conv output was silently the wrong length (it included the cache-prepended
  warm-up columns transformers itself trims away one call layer up) and was mislabeled with a
  `_prefill` suffix regardless of stage. `gdn_conv_out_fn_full`/`gdn_conv_out_fn_trimmed` now make
  both forms available, with `_trimmed` matching `Qwen3_5GatedDeltaNet.forward`'s own external
  slice.
- Post-rope q/k (`attn_q_post_rope`/`attn_k_post_rope`), the sigmoid gate (`attn_gate_sigmoid`),
  and the pre-`o_proj` post-gate tensor (`attn_post_gate`) are now captured for full-attention
  layers; previously only pre-rope `attn_q_normed`/`attn_k_normed` existed despite the docs
  claiming post-rope coverage.
- `tests/reference/test_manifest.py` is now a pytest-free script (bare asserts + `__main__`) that
  actually runs (`python tests/reference/test_manifest.py`, no venv install needed) and is
  registered as ctest test `reference_manifest` in `tests/CMakeLists.txt` -- previously its 20
  assertions had only ever been driven by a manual throwaway harness, never by a live test run, and
  it wasn't wired into any runner.
- `GdnCapture` now asserts its required tensors were actually recorded and raises (surfacing as
  `status="error"` in the manifest) rather than silently writing a golden with the GDN intermediates
  missing, in case an optional `kernels`/`fla`/`causal_conv1d` hub package rebinds
  `Qwen3_5GatedDeltaNet`'s module-level functions out from under the monkeypatch.
- `--force-random-init` mode now perturbs every `Qwen3_5RMSNorm`/`Qwen3_5RMSNormGated` weight with
  small seeded noise (`transformers`' own default init is exactly zero, which made the norm's
  `(1 + weight)` scale identically `1.0` and unable to catch a missing/incorrect weight multiply).

## Rung 4 -- full logits, offloaded, few tokens

**Not yet measured.** Run the *whole* model (both `transformers`, CPU-offloaded a layer at a time
given the 27B size, and r4dx end to end on HIP device 1) on real prompt tokens and diff final
logits. This is where rung 3's per-layer tolerances compound -- expect a visibly looser bound than
any single layer's, and a plan for *how much* looser (e.g. `rel_err` growing roughly with
`sqrt(num_layers)` for independent per-layer error, `2e-2 * sqrt(64) ~ 0.16` as a first guess)
needs to be pinned down empirically, not guessed here.

The r4dx half of the **tooling** for that measurement is built (2026-09-22) -- see "Rung 4 tooling"
below. The `w4a16` measurement itself is in "Rung 4 measurement: w4a16" further down.

### Rung 4 tooling

Rung 4 compares distributions, not point predictions, so the artifact both sides produce is a
**teacher-forced per-position next-token log-probability dump**: for a FIXED token sequence
`ids[0..T-1]`, `rows = T-1` vectors where

```
row i = log_softmax(logits at position i) = log p(next token | ids[0..i]),   i in [0, T-2]
```

Nothing is sampled; `ids[T-1]` is never fed (no row would read its logits). Both halves write the
same format so a KL report can pair them row for row:

| file | contents |
|---|---|
| tokens file (JSON) | `{"tokenizer": "<hf path or name>", "segments": [{"name": "<str>", "token_ids": [int, ...]}, ...]}` -- ids from the checkpoint's own HF `AutoTokenizer` with `add_special_tokens=False` on the raw text: **no chat template, no BOS**. Each segment is scored independently from a fresh context at position 0. |
| `<out-dir>/<segment>.logprobs.f16` | raw little-endian float16, row-major `[T-1, V]`, no header. `V` is the model's own `lm_head` vocab (`Config().vocab_size`), which the tools print rather than assume. |
| `<out-dir>/<segment>.meta.json` | `{"T", "V", "dtype": "float16", "rows": T-1, "source": "<r4dx\|reference>", "layout"\|"torch_dtype", "sha256_of_token_ids_json"}` (plus provenance extras). |

`sha256_of_token_ids_json` is the SHA-256, lowercase hex, of the **compact** JSON array of that
segment's ids -- `[1,2,3]`, UTF-8, no spaces, no trailing newline, i.e.
`json.dumps(ids, separators=(",", ":")).encode("utf-8")`. It is what lets the report prove the two
halves scored the same tokens. The two implementations are
`tools/reference/make_tokens_json.py`'s `token_ids_sha256()` and `r4dx_tf::TokenIdsSha256`
(`tests/model/teacher_forced.h`); ctest's `test_teacher_forced_logprobs` pins the C++ one with the
FIPS 180-4 `"abc"` known answer.

**fp16 clamp.** fp16 cannot hold a log-prob below about -65504 and loses all resolution long
before that, so every value is clamped to `-1e4` before the cast (in fp32, after the log_softmax).
Every token this touches has probability below `e^-10000` -- exactly zero in any arithmetic either
side can do -- so no KL sum moves measurably. It is recorded as `clamp_min` in the r4dx sidecar.

**The r4dx half: `tests/model/tool_teacher_forced_logprobs`** (source
`tests/model/tool_teacher_forced_logprobs.cpp`, pass in `tests/model/teacher_forced.h`). Built like
the other `tests/model` tools (`tool_hseed_drift`, `tool_dflash_probe`, ...) and, like them,
deliberately **not** registered with `add_test()`.

```powershell
$env:HIP_VISIBLE_DEVICES = '1'
build\win-hip\tests\model\tool_teacher_forced_logprobs.exe `
    --model D:/models/r4dx/qwen38-27b-v3.r4dx --layout w4a16 `
    --tokens <tokens.json> --out-dir <dir> [--segment <name>] [--max-ctx N] [--layers N] `
    [--vision off] [--check-greedy N] [--no-write] [--quiet]
```

It loads the container once and drives the sequence through the engine's **own decode path**:
`Model::Prefill({ids[0]})` for row 0, then `Model::DecodeStep(ids[i])` for row `i` -- the same GDN
recurrent-update kernels, the same paged fp8 KV cache with its calibrated descales, and the same
fused quant epilogues a real generation uses for the positions it generates. MTP and DFlash2 are
unconditionally off (both are strategies for *generating*, and this tool never generates), which
also keeps `Model`'s draft window at 1, i.e. the pre-speculation GDN state layout. `--vision`
defaults to **off** here: the tower plays no part in a text-only log-prob dump and skipping it
reclaims ~0.89 GiB. `--layers N` exists for the 4-layer test containers, whose `config.json` still
declares the full 64.

What it is NOT: the chunked-**prefill** path. A real generation runs its prompt through the
chunked-scan GDN kernels (one `Prefill` call over many tokens) and only its generated positions
through the per-token path. Feeding the whole sequence one token at a time is what makes every row
comparable to every other row and to the reference's own uniform pass; the price is that the single
row sitting on a prompt/generation boundary is computed here by the decode path where the
generation computed it by the prefill path. Measured on the 4-layer container, the two agree to
~7e-4 in row logsumexp against a neighbouring-row spread of ~1e-1, i.e. two orders of magnitude of
separation -- enough that nothing downstream confuses the two, but not bit-for-bit, which is why
the automatic test below excludes exactly that one row from its argmax assertion.

**Numerics.** The row reduction is `m = max_j logits[j]`, `S = sum_j exp(logits[j] - m)`,
`lse = m + log(S)`, `lp[j] = logits[j] - lse`, all on the host from the engine's fp32 logits, with
`S` accumulated in `double` and the result rounded back to fp32. That is not a deviation from
"computed in fp32": a naive fp32 sum over 248320 terms loses several digits to cancellation, while
torch's own fp32 `log_softmax` uses a blocked/pairwise reduction much closer to the exactly-rounded
answer -- accumulating in double lands on the same fp32 value the reference does rather than a
different one for a reason that has nothing to do with the model. The tool always checks its own
output is a distribution (`max |log sum_j exp(row[j])| < 1e-2`, measured ~1.5e-6) and exits 1 if not.

**Getting the token ids right.** Two producers, both writing the tokens file format above:

- `tools/reference/make_tokens_json.py` -- text files through the checkpoint's `AutoTokenizer`,
  one segment per file, with per-segment truncation. This is what a corpus-driven KL run uses. Runs
  in the read-only reference venv; no GPU, no weights, only the tokenizer files.
- `r4dx-cli --dump-token-ids <tokens.json>` -- writes the ids **actually fed into the model's
  KV/GDN state** so far (the rendered prompt plus every committed generated token) as a single
  `"cli"` segment, rewritten in full after every turn. Re-tokenizing the printed text cannot
  reproduce them: the chat template's special tokens, and anything the stream decoder's
  `skip_special_tokens=true` dropped, would be lost. This is what makes a "generate, then score what
  you generated" check possible at all.

**The automatic check: ctest `test_teacher_forced_logprobs`**
(`tests/model/test_teacher_forced_logprobs.cpp`, HIP device 1, SKIPPED when
`D:/models/r4dx/qwen38-27b-l4-mtp.r4dx` is absent). It drives the *same* pass -- both TUs call
`teacher_forced.h`, so the tool's numbers are the ones ctest checks, not a lookalike
reimplementation -- and asserts three things on the 4-layer container:

1. **Greedy consistency.** Generate 64 tokens greedily from a fixed prompt, teacher-force the whole
   prompt+generated sequence, and require `argmax(row i) == ids[i+1]` for every row predicting a
   generated token (except the prefill/decode boundary row above). Catches an off-by-one position,
   the wrong row being read, or the sequence being fed into a different state than generation used.
2. **Normalization.** `|log sum_j exp(row[j])| < 1e-2` for every row.
3. **Row alignment against the generation loop's own logits.** Check 1 is only as strong as the
   generated tail is varied, and a 4-layer truncation of a 27B model collapses onto one repeated
   token, against which an off-by-one row would still "pass". So the test also compares each shared
   row's logsumexp -- one scalar depending on all `V` logits -- between the dump and the decode loop
   that generated the sequence, and additionally asserts that *neighbouring* rows' logsumexps are
   separated from that disagreement by at least 10x, so the check can never go quietly vacuous.
   Measured: disagreement 7.2e-4, neighbour spread 1.0e-1, 63/63 rows distinct.

**Not done here, deliberately.** A `Model::VerifyWindow`-batched fast path would cut the ~26 ms per
row on the real container by processing several positions per forward pass, but it cannot be shown
*row-identical* to the step path: a `T=8` verify window and a `T=1` decode step run different GEMM
shapes and therefore different reduction orders, so the two agree to reduction-order noise, not bit
for bit. A dump whose rows depend on an internal batching knob is the wrong artifact to compute a
KL divergence from, so the tool has one path. At ~26 ms/row a 2048-token segment costs about a
minute, which is not the bottleneck in this rung.

### Rung 4 measurement: w4a16

**Measured 2026-09-22, HIP device 1, real 64-layer container `D:/models/r4dx/qwen38-27b-v3.r4dx`.**

**Corpus**: `tools/reference/kl_corpus/` (committed), the held-out four-segment corpus described in
`tools/reference/README.md` -- `english_prose` (original), `cpp_source` (excerpt of this repo's own
`src/model/model.cpp`), `python_source` (excerpt of this repo's own `tools/reference/layer_golden.py`),
`thai_prose` (original) -- each truncated to exactly 1024 tokens in the committed `tokens.json`.
Disjoint from `tools/reference/calib.txt`, which chose the fp8 KV descale constants.

**Exact commands** (both sides on HIP device 1, one GPU process at a time):

```powershell
$env:HIP_VISIBLE_DEVICES = '1'
$py = "<reference venv>\Scripts\python.exe"   # see tools/reference/README.md

# reference: bf16 checkpoint, all 4 segments
& $py tools\reference\full_logits_golden.py --device cuda `
    --tokens tools\reference\kl_corpus\tokens.json --out-dir tools\reference\kl_out\ref

# r4dx: w4a16, all 4 segments
build\win-hip\tests\model\tool_teacher_forced_logprobs.exe `
    --model D:/models/r4dx/qwen38-27b-v3.r4dx --layout w4a16 `
    --tokens tools\reference\kl_corpus\tokens.json `
    --out-dir tools\reference\kl_out\w4a16 --max-ctx 4096 --vision off

# report
& $py tools\reference\kl_report.py `
    --ref-dir tools\reference\kl_out\ref --test-dir tools\reference\kl_out\w4a16 `
    --tokens tools\reference\kl_corpus\tokens.json --out tools\reference\kl_out\kl_w4a16.json
```

**Wall time / VRAM.** Reference: ~24.7-42.0s per 1024-token segment (`stack` time; the first
segment of a run pays a cold page cache), peak 1.56 GiB allocated. r4dx `w4a16`: 28.7-28.8 ms/row
(~29.3-29.5s per 1023-row segment), 117.4s total for 4 segments, peak **16.23 GiB** VRAM (this run
did not share device 1 with another process). Both sides ran once each, sequentially, never
concurrently.

**Result** (`KL(P_ref || Q_w4a16)` in nats, fp64, full 248320-way vocabulary, per position):

| segment | rows | mean KL | median KL | p99 KL | max KL | max @ | top-1 | top-5 | ppl ref | ppl w4a16 | KL>1 |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| cpp_source | 1023 | 0.05907 | 0.02025 | 0.46623 | 1.51422 | pos 662 (tok 9 `'*'`) | 91.01% | 99.80% | 3.805 | 3.921 | 2 |
| english_prose | 1023 | 0.06765 | 0.04152 | 0.34000 | 0.73583 | pos 109 (tok 82) | 89.74% | 99.80% | 3.755 | 4.024 | 0 |
| python_source | 1023 | 0.06869 | 0.03267 | 0.40826 | 0.95169 | pos 645 (tok 13011) | 89.05% | 99.51% | 4.605 | 4.922 | 0 |
| thai_prose | 1023 | 0.15633 | 0.10060 | 0.83319 | 3.89099 | pos 206 (tok 53900 `'์'`) | 78.20% | 98.53% | 15.610 | 16.991 | 8 |
| **ALL** | 4092 | **0.08794** | 0.04833 | 0.58740 | 3.89099 | thai_prose pos 206 | **87.00%** | 99.41% | 5.661 | 6.027 | 10 |

**Positions with KL > 1 nat** (10 total, all in code/Thai; none in `english_prose`/`python_source`):

- `cpp_source`: pos 526 (next token 874, `' no'`, KL 1.288), pos 662 (next token 9, `'*'`, KL 1.514).
- `thai_prose`: pos 206 (tok 53900 `'์'`, KL 3.891), pos 257 (tok 38534 `'ต'`, KL 1.351), pos 258
  (tok 148410 `'้อง'`, KL 2.035), pos 318 (tok 149334 `'ญา'`, KL 1.717), pos 636 (tok 157620
  `'เฉล'`, KL 2.376), pos 669 (tok 45596 `'ี่'`, KL 1.290), pos 949 (tok 35982 `'ว'`, KL 1.887), pos
  962 (tok 148947 `'ูก'`, KL 1.808). Every Thai KL>1 token is a sub-syllable script fragment, not a
  whole word -- consistent with the tokenizer's byte/character-level fallback for Thai and with the
  corpus's much higher baseline reference perplexity (15.6 vs 3.8-4.6 for the other three), rather
  than any obviously localized bug. Not investigated further per this stage's scope (report, don't
  fix).

**The `-1e4` clamp caveat.** Every logprob below -1e4 is floored to -1e4 in fp32 before the fp16
cast on both sides (`exp(-1e4) == 0.0` even in fp64), so the KL sum is unaffected by how far below
that floor a token's true log-prob sits. The per-segment fp16 round-trip diagnostic
(`max |sum_v exp(logp_ref[v]) - 1|`, expect ~1e-3) came back `1.13e-03` / `8.84e-04` / `9.59e-04` /
`1.02e-03` for the four segments -- nowhere near ~1, so the clamp/format is not corrupting the
distributions.

**Sanity controls** (per this stage's brief):

1. **Reference vs itself.** `kl_report.py` with `--ref-dir`/`--test-dir` both pointed at
   `tools/reference/kl_out/ref` gives exactly **0.00000** mean/median/p99/max KL and **100.00%**
   top-1/top-5 on all four segments -- the report does not manufacture disagreement out of nothing.
2. **Deliberate wrong pairing.** `cpp_source`'s reference dump paired against a file *named*
   `cpp_source.logprobs.f16` but containing `thai_prose`'s reference bytes (copied into a scratch
   dir under the mismatched name, `--allow-mismatch` since the sidecar's own sha256 was left as
   `cpp_source`'s -- the point is to defeat only the content check, not the file-naming
   convention): **mean KL 16.883 nats**, top-1 agreement **0.20%**, ppl_test **~6.97e7**, 1022/1023
   positions KL>1 -- three orders of magnitude above the real `w4a16` result, proving the report
   is not trivially returning small numbers.
3. **Independent recomputation.** A from-scratch 10-line numpy script (no `kl_report` import) that
   memory-maps both `cpp_source.logprobs.f16` files whole, widens to fp64, and computes
   `mean(sum_v exp(ref) * (ref - test))` directly gave **0.059074700681517434**, against
   `kl_report.py`'s own **0.059074700681517114** -- agreement to `3.2e-16`, far inside the requested
   `1e-6`.

**Interpretation** (stated as guidance, not a verdict this document draws for the reader):
llama.cpp `Q4_K_M`-class quantizations typically land at mean KL ~0.02-0.05 nats vs bf16 with top-1
agreement in the mid-90s%. Rule of thumb: mean KL < 0.1 = close, > 0.3 = something is broken. This
run's overall mean KL is **0.08794** (top-1 87.0%) -- inside the "close" band but above the
`Q4_K_M` range and with `thai_prose` alone at 0.156 mean / 78.2% top-1, i.e. visibly worse than the
other three segments. `w4a16`'s ppl is consistently ~3-9% above the bf16 reference on every
segment (3.921/3.805, 4.024/3.755, 4.922/4.605, 16.991/15.610), a uniform small perplexity
inflation rather than one segment breaking outright.

## Rung 5 -- generation sanity

Built (`src/cli`/`r4dx-cli.exe`) and, as of the 2026-09-20 long-context validation pass
(docs/r9700.md R13 + Q17), exercised up to the model's own native 262144-token context ceiling, not
just "hundreds to thousands of tokens" -- real prompts (docs/perf.md's standard haiku prompt, and a
needle-retrieval prompt with a distinctive fact stated near the start of the document and asked for
at the end) padded with real corpus text to reach 2k/8k/32k/131072/262144 tokens, through the real
chat template, chunked prefill, and paged KV cache. Result: coherent English, no repetition
collapse, no garbage tokens, and correct needle recall at every context length tested, including
262144 real prefilled tokens -- this is exactly the "off-by-one in paged block indexing / wrong
RoPE section boundary / silently-zeroed GDN state" class of bug this rung exists to catch, and it
did not fire at any tested context length. `--mtp 3` reproduced byte-identical text to `--mtp 0` at
every context length, an additional cross-check that MTP's window bookkeeping stays correct this
far out too. Full data, exact prompts, and transcripts: `docs/perf.md`'s "Long-context validation"
section. Not yet done: a systematic needle-position sweep (fact at 10%/50%/90% depth, multiple
distinct facts) -- this pass used one prompt design per context length, sufficient to show position
handling works at all, not a full needle-in-a-haystack accuracy curve.

### Rung 5 for images (2026-09-22, `docs/vision.md` "End to end: does the model actually see the
picture")

The vision path's version of the same question, and it needs a different kind of prompt than a
photograph: a picture a human judges "looks about right" cannot separate a correct splice from a
subtly wrong one. The four cases used all have an answer known BY CONSTRUCTION -- three of them
rendered locally with `System.Drawing` from PowerShell (no network, so they are reproducible on
this box and carry no licence question): a known string for OCR, a chart with known colours and a
known tallest bar, and an image with a known number of countable objects, plus the tower golden's
own synthetic gradient/checkerboard/circle image. Greedy decoding throughout, on the real
container, through `tests/vision/tool_vision_chat`. All four answered correctly; prompts and
verbatim answers are in `docs/vision.md`.

This is the rung that catches the "positions look plausible but are shifted" class -- the failure
mode of a wrong mrope advance rule or a mis-offset splice is fluent, confident nonsense about a
different picture, which no tolerance test upstream would flag. It also distinguishes that class
from plain resolution limits: the OCR case was run at three render sizes and the error is monotone
in the patch grid (`HELLO`/`RDX 791` at a 20x40 grid, `HELLO R4DX 7301` at 40x80, exact at a 20x80
grid with one line), which is a resolution curve, not a bookkeeping bug.

Not yet done at this rung: a real photograph (all four images are synthetic or rendered).
`image_url` in the server's own request path is DONE (2026-09-22, stage 5) -- see
`docs/server.md`'s "Images" and `docs/vision.md`'s "User-facing wiring", including a real
`tools/server/smoke.ps1 -Vision` run against the real container: a rendered-text OCR image read
back exactly, a synthetic shapes image described correctly, and an image-aware prefix cache
verified to reuse an already-fed image's rows (no re-encode) while never reusing a different
image's KV state at the same conversation position.
