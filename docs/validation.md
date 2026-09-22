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

`tools/reference/kv_calibrate_full.py` is the same rung's sibling for the fp8 KV descale tables:
per-kv-head `amax` of K (post-rope) and V at all 16 full-attention layers, which the converter
turns into `text.layers.{i}.attn.k_descale`/`.v_descale` (`amax / 448.0`, fp8 e4m3 max). It runs
the **real** full 64-layer bf16 stack over a 6-file / 8316-token calibration corpus (reusing
`full_logits_golden.py`'s layer-streaming forward), so the K/V it measures come from the true
mid-stack activations, and writes all 16 layers to one merged JSON. Its own gates -- the K tap
proven post-rope by a norm-preservation check, run-to-run stability, and the comparison against the
prototype -- are in `tools/reference/README.md`, "kv_calibrate_full.py".

`tools/reference/kv_calibrate.py` is the **prototype** that preceded it (see its `README.md`
section and its output JSON's own `"caveat"` field): it feeds raw token embeddings straight into
one layer, skipping every preceding layer, so it calibrates against a distribution that layer never
sees. It pinned down the JSON contract and the descale formula, and measured against the full
forward it runs 1.4-3.3x low on `k_amax` and up to 8.8x low on `v_amax`. Do not convert a shipping
container with it.

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
- ~~**`kv_calibrate.py`'s calibration distribution is not representative**~~ -- **CLOSED** by
  `tools/reference/kv_calibrate_full.py` (full 64-layer forward, all 16 layers, 8316 tokens of
  mixed English/Thai/C++/Python/chat-template corpus), which writes
  `D:\models\r4dx\qwen38-27b.kvcalib-full.json`. What remains open is downstream, not here: the
  shipped container `D:\models\r4dx\qwen38-27b.r4dx` still carries the **prototype's** descales and
  has to be re-converted with `--kv-calib D:\models\r4dx\qwen38-27b.kvcalib-full.json` before its
  fp8 KV cache is trustworthy. Note the remaining caveat that no calibration pass can close: these
  are **static** per-kv-head scales, fixed at convert time, so an activation above the corpus's
  amax saturates at +-448 rather than getting its own scale. The JSON's `k_p9999`/`v_p9999` fields
  are there to keep the size of that tail visible.

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

**Measured for `w4a16` on 2026-09-22, and audited the same day.** Run the *whole* model (both
`transformers`, streamed a layer at a time given the 27B size, and r4dx end to end on HIP device 1)
over real token sequences and compare final distributions. This is where rung 3's per-layer
tolerances compound, so the artifact is a **distribution** comparison (per-position KL over the full
vocabulary) rather than a per-element `rel_err` bound: the original guess here was that `rel_err`
would grow like `sqrt(num_layers)` (`2e-2 * sqrt(64) ~ 0.16`), which is not a quantity a quantized
engine can usefully be held to at the logit level.

Three sections follow: **"Rung 4 tooling"** (the shared format and both halves' programs),
**"Rung 4 measurement: w4a16"** (the numbers), and **"Auditing this measurement"** (the adversarial
re-examination of those numbers, and the two tools that make it re-runnable). Layouts other than
`w4a16` -- `w4a8`, `mxfp4` -- have not been measured yet; the tooling takes `--layout`, so each is
one tool invocation plus one `kl_report.py` run against the same reference dumps.

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
  whole word. The `thai_prose` outlier is decomposed in "Auditing this measurement" below; it is
  real quantization loss, not a tokenizer or corpus artifact.

**What this number is a delta of.** The whole `w4a16` *container as configured*, not the body
weights alone: 4-bit body weights with fp16 activations, a **4-bit `lm_head`**, and the **fp8 paged
KV cache** with its calibrated descales, all driven through the engine's per-token decode kernels.
Attributing the 0.088 between those three is a separate experiment (the container carries no bf16
body layout, so it cannot be done by re-running this tool with a different `--layout`).

**The `-1e4` clamp caveat -- measured, and it is a no-op.** Every logprob below -1e4 is floored to
-1e4 in fp32 before the fp16 cast on both sides. `kl_audit.py` check 3 counts how often that
actually fires: across all eight dumps (4 segments x 2 sides, `1023 x 248320` entries each)
**zero** entries were at or below the floor, on either side, and therefore zero reference
probability mass sits on a clamped test entry. The floor is insurance against an fp16 `-inf`, not
a term in this result. The per-segment fp16 round-trip diagnostic
(`max |sum_v exp(logp_ref[v]) - 1|`, expect ~1e-3) came back `1.13e-03` / `8.84e-04` / `9.59e-04` /
`1.02e-03` for the four segments -- nowhere near ~1, so the format is not corrupting the
distributions either.

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

### Auditing this measurement

**Audited 2026-09-22, HIP device 1.** The numbers above survived an adversarial re-examination whose
premise was that they are wrong. Two tools were written for it and are committed, so every check
below is re-runnable rather than a one-off:

- `tools/reference/kl_audit.py` -- numpy only, no GPU, no checkpoint. Runs on the same
  `--ref-dir`/`--test-dir`/`--tokens` triple as `kl_report.py` and adds five checks: identity,
  alignment (the off-by-one control), clamp accounting, KL-vs-reference-entropy, and the KL split by
  vocabulary region. `--self-test` plants a deliberately one-row-shifted dump with a clamped column
  and requires the audit to report exactly that.
- `tools/reference/reference_selfcheck.py` -- GPU, validates the *reference half* against code it
  does not share.

```powershell
$env:HIP_VISIBLE_DEVICES = '1'
& $py tools\reference\kl_audit.py --self-test
& $py tools\reference\kl_audit.py `
    --ref-dir tools\reference\kl_out\ref --test-dir tools\reference\kl_out\w4a16 `
    --tokens tools\reference\kl_corpus\tokens.json --out tools\reference\kl_out\kl_audit_w4a16.json
& $py tools\reference\reference_selfcheck.py --tokens tools\reference\kl_corpus\tokens.json `
    --truncated 4 --tokens-n 48 --noise-floor --noise-n 256 `
    --out tools\reference\kl_out\reference_selfcheck.json
```

**1. Is the reference itself faithful?** This is the check the measurement most depends on and the
one `full_logits_golden.py`'s own `--cross-check` *cannot* make: `--impl model` and `--impl manual`
share `build_layer`, `gather_embedding_rows` and `logits_fp32`, so a bug in the streaming machinery
is invisible to them. `reference_selfcheck.py --truncated 4` therefore builds a genuine, fully
resident `Qwen3_5TextModel` with `num_hidden_layers = 4` -- ordinary `nn.Module`, ordinary
`load_state_dict` from the same shards, ordinary `nn.Linear` over the whole 248320-way lm_head --
and compares it to `StreamingReference(max_layers=4)` at **every** position. `layer_types[:4]` is
`[linear, linear, linear, full]`, so both the GDN and the full-attention path are covered.

| streaming impl | hidden max\|diff\| | logits max\|diff\| | argmax agreement | per-position KL(control‖streaming) |
|---|---:|---:|---:|---:|
| `model` | 2.99e-01 | 3.13e-01 | 48/48 | max 1.74e-03, mean 3.42e-04 |
| `manual` | 2.99e-01 | 3.13e-01 | 48/48 | max 1.74e-03, mean 3.42e-04 |

The hidden-state disagreement is 2.99e-01 against `|hidden|max = 50.5`, where **one bf16 ulp is
1.97e-01** -- i.e. one to two ulps, which is what two differently-ordered bf16 reductions of the
same arithmetic are entitled to, and there is no drift with position. The streaming composition is
faithful.

**2. The reference's own noise floor at full depth.** `--noise-floor` runs all 64 layers through
both compositions over 256 positions and measures the per-position KL between them: **mean
3.94e-04**, median 3.3e-05, p99 3.5e-03, max 5.3e-03, top-1 self-agreement **99.61%** (a second run
gave mean 3.47e-04 / 99.22%, which is the run-to-run spread of GPU reduction order). So the
reference disagrees with *itself* by ~0.4% of the 0.08794 being reported, and agrees with itself on
the top token 99%+ of the time against the 87.00% measured for `w4a16`. Neither the measured KL nor
the measured top-1 gap is bf16 noise.

**3. Is the pairing aligned?** An off-by-one on either side is the failure that looks like "merely a
large KL". `kl_audit.py` check 2 pairs the rows deliberately wrong and reports both:

| segment | top-1 aligned | top-1 `ref[i]`/`test[i+1]` | top-1 `ref[i+1]`/`test[i]` | mean KL aligned | mean KL shifted +1 |
|---|---:|---:|---:|---:|---:|
| cpp_source | 91.01% | 2.84% | 2.84% | 0.05907 | 15.108 |
| english_prose | 89.74% | 3.42% | 3.33% | 0.06765 | 13.225 |
| python_source | 89.05% | 2.05% | 2.64% | 0.06869 | 14.049 |
| thai_prose | 78.20% | 1.66% | 1.47% | 0.15633 | 9.125 |

A misaligned dump would score the right-hand columns. It scores the left-hand ones. Independently,
the reference's own argmax hits the corpus's actual next token 71.95% / 63.54% / 66.67% / 41.25% of
the time -- a sane profile for a 27B model on code / English / Python / Thai, and impossible if row
`i` were not predicting token `i+1`. Both sidecars' `sha256_of_token_ids_json` also recompute to the
tokens file's own hash on all four segments, so the two halves provably scored the same ids.

**4. Is the dumped row what the engine actually samples from?** `r4dx-cli` generated 64 tokens
greedily on the **real 64-layer `w4a16` container** (`--temperature 0`, `--vision off`, no MTP, no
DFlash) writing `--dump-token-ids`, and `tool_teacher_forced_logprobs --check-greedy 64` then scored
that exact 92-token sequence: **64/64 rows' argmax equals the token the CLI committed**, including
the prompt/generation boundary row, at `max |logsumexp(row)| = 1.7e-06`. The dump is the generation
path's own distribution, not a lookalike. (`--vision off`, `mtp 0`, `dflash false` are recorded in
each r4dx sidecar; `tool_teacher_forced_logprobs.cpp` hard-codes the last two.)

**5. Where does the divergence actually come from?** Two decompositions, both in `kl_audit.py`.

*By reference entropy* (check 4). KL grows monotonically with how flat the bf16 distribution already
is, in every segment -- from ~0.005 nats where the reference is near-certain (`H < 0.25`) to
~0.17-0.25 where it is spread wide (`H > 3`). `thai_prose` has 72.7% of its positions above `H = 1`
and 33.9% above `H = 3`, against 39-51% and 4.8-16.1% for the other three. Reweighting each segment
onto `cpp_source`'s entropy histogram:

| segment | raw mean KL | entropy-matched mean KL |
|---|---:|---:|
| cpp_source | 0.05907 | 0.05907 |
| english_prose | 0.06765 | 0.05551 |
| python_source | 0.06869 | 0.05837 |
| thai_prose | 0.15633 | 0.08930 |

So the other two ASCII segments are, at matched confidence, slightly *better* than `cpp_source`, and
`thai_prose`'s 2.65x raw gap shrinks to 1.51x. Roughly 60% of the Thai outlier is composition -- the
bf16 model is simply far less certain there (mean entropy 2.26 vs 1.15-1.36 nats; reference
perplexity 15.6 vs 3.8-4.6) -- and ~40% is a genuine Thai-specific excess.

*By vocabulary region* (check 5), which is what that remaining 40% is. `thai_prose` draws **50.7% of
its token ids from `>= 148000`**, the checkpoint's extended/multilingual vocabulary tail, against
0.0% / 0.0% / 0.2% for the other three. Splitting the KL sum there:

| segment | ref mass in ids ≥148000 | KL from that region | KL per unit mass, low / high | mass-weighted \|Δ logp\|, low / high |
|---|---:|---:|---:|---:|
| cpp_source | 0.00026 | -0.00003 (-0.1%) | +0.0591 / -0.1237 | 0.174 / 0.462 |
| english_prose | 0.00011 | -0.00000 (-0.0%) | +0.0677 / -0.0086 | 0.223 / 0.464 |
| python_source | 0.00046 | -0.00007 (-0.1%) | +0.0688 / -0.1541 | 0.207 / 0.552 |
| thai_prose | 0.60318 | +0.12362 (**79.1%**) | +0.0824 / **+0.2049** | 0.318 / 0.426 |

The last column is the tell: in **every** segment, including the three that are 100% ASCII, the
quantized model reproduces log-probabilities in the `>= 148000` tail **2.1-2.7x** less accurately
than in the head of the vocabulary. For the ASCII segments that region holds ~0.01-0.05% of the
probability mass, so it costs them nothing; for `thai_prose` it holds 60% of the mass and supplies
79% of the KL, at 0.205 nats per unit mass against 0.082 in the low region. The Thai excess is the
4-bit `lm_head`'s rare-token rows being reconstructed worse, surfacing on the one segment whose
predictions live there -- a property of the quantization, not of the measurement.

*Not a corpus or tokenizer artifact.* All four corpus files are already NFC-normalized,
`decode(tokenize(text))` is a prefix of the source text for each, re-tokenizing reproduces the
committed ids exactly, and no line longer than 20 characters is shared with `calib.txt` (the fp8 KV
descale calibration corpus), so the corpus is genuinely held out.

**Verdict.** Trustworthy as stated: the KL is `KL(P_bf16 ‖ Q_w4a16)`, in nats, over the full
248320-way vocabulary, accumulated in fp64, averaged over positions, on aligned rows of identical
token ids, against a reference validated against a plain transformers forward, with a reference
self-noise ~200x smaller than the reported value.

### Follow-up experiments (2026-09-22): where the 0.088 comes from

Same corpus, same reference, same tooling; every run is `tool_teacher_forced_logprobs` + `kl_report`
(outputs under `tools/reference/kl_out/<name>/`, JSON summaries `kl_<name>.json`). Mean KL is the
mean over the four segments; PPL is the corpus perplexity of the r4dx side (reference 5.661).

| container / layout | mean KL | cpp | english | python | thai | top-1 | PPL | runtime cost |
|---|---|---|---|---|---|---|---|---|
| v3 w4a16 (prototype KV calib, 4-bit lm_head) -- the rung 4 number | 0.088 | 0.059 | 0.068 | 0.069 | 0.156 | 87.0% | 6.03 | -- |
| v3 mxfp4 | 0.111 | 0.070 | 0.085 | 0.088 | 0.204 | 84.9% | 6.28 | -- |
| v3 w4a8 | 0.159 | 0.106 | 0.128 | 0.124 | 0.278 | 82.5% | 6.72 | -- |
| w4a16 + **bf16 lm_head** (`--lm-head bf16`, prototype KV calib) | 0.079 | 0.053 | 0.059 | 0.061 | 0.143 | 87.2% | 5.92 | +1.9 GiB VRAM, ~-6% decode |
| w4a16 + **full-forward KV calib** (`kv_calibrate_full.py`, 4-bit lm_head) | **0.072** | 0.045 | 0.057 | 0.052 | 0.136 | **88.4%** | 5.85 | **none** |

What this says:

- **The weight quantization dominates.** Three layouts sharing the same fp8 KV cache and the same
  4-bit `lm_head` spread 0.088 -> 0.159; if the KV path were the floor they would cluster. w4a8's
  `wq` bytes are identical to w4a16's, so its extra +0.07 nats is the price of per-row int8
  activation quantization alone.
- **The `lm_head` is not where the vocab-tail loss lives.** A bf16 head recovers only ~0.009 nats
  and barely moves the >=148000 vocab-region error (`kl_audit.py` check 5: thai mass-weighted
  |dlogp| 0.426 -> 0.400), which corrects the attribution in "Auditing this measurement" above: the
  tail tokens are simply where the body's error is most visible. Not worth 1.9 GiB.
- **The prototype KV calibration was clipping.** `kv_calibrate_full.py` (real mid-stack activations,
  8316 tokens, six files incl. a chat-templated turn) finds K amax ~2x and V amax up to ~8x larger
  than the prototype's, i.e. the old descales saturated the e4m3 cache on ordinary text. Fixing the
  calibration alone is worth -18% KL and +1.4 points top-1 at zero runtime cost, and every new
  container should be converted with `--kv-calib D:\models\r4dx\qwen38-27b.kvcalib-full.json`.
- Published llama.cpp Q4_K_M on this checkpoint measures 0.011-0.014 mean KL / 95-96% top-1 (its
  own `llama-perplexity --kl-divergence` methodology, wikitext / held-out English). The remaining
  gap is in `quant_int4.hpp`'s grid: plain min/max scale, round-to-nearest, no scale/zero search,
  no importance weighting, 128-element groups -- the first two are converter-only changes with no
  runtime cost and are the next step.

Converter/loader changes made for this pass: `--lm-head bf16` now survives `--no-bf16` (only the
default lm_head spec is stripped), and `Container::Load` falls back to `lm_head.bf16.w` when the
requested layout is absent from the container.

### Milestone 10: grid search + imatrix (2026-09-22)

Two full 64-layer w4a16-only containers converted with the Stage-2 `quant_search.hpp` sweep
(`--quant search`, 21-step 0.85x-1.15x multiplier grid, never-worse-than-RTN), both `--mtp on
--vision on --no-bf16 --lm-head 4bit`, same full-forward KV calibration as the baseline:

- **A** `D:\models\r4dx\qwen38-27b-w4a16-search.r4dx` -- `--quant search`, no imatrix. Converted in
  **250.996 s**.
- **B** `qwen38-27b-w4a16-search-imatrix.r4dx` (`--quant search --imatrix
  D:\models\r4dx\qwen38-27b.imatrix.npz`) -- **not built this stage**, see "What's missing" below.

Baseline for both is the existing `qwen38-27b-w4a16-kvfull.r4dx` (`--quant rtn`, same KV calib,
already measured: `tools/reference/kl_out/w4a16-kvfull`, mean KL 0.072 / top-1 88.4%).

**Full KL table, container A** (`tools/reference/kl_out/w4a16-search`, `kl_w4a16-search.json`):

| segment | rows | mean KL | median KL | p99 KL | max KL | top-1 | top-5 | ppl test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpp_source | 1023 | 0.04905 | 0.01564 | 0.43222 | 1.78530 | 91.10% | 99.80% | 3.958 |
| english_prose | 1023 | 0.05096 | 0.02828 | 0.28884 | 1.49641 | 89.44% | 99.90% | 3.992 |
| python_source | 1023 | 0.05261 | 0.02455 | 0.31608 | 0.85268 | 90.13% | 99.90% | 4.817 |
| thai_prose | 1023 | 0.13259 | 0.09048 | 0.81995 | 2.07074 | 80.16% | 98.92% | 17.007 |
| **ALL** | 4092 | **0.07130** | 0.03661 | 0.53250 | 2.07074 | **87.71%** | 99.63% | 5.998 |

**Unweighted search essentially matches the RTN baseline, it does not beat it**: 0.0713 mean KL /
87.71% top-1 vs RTN's 0.072 / 88.4% -- within noise on mean KL, and top-1 is actually *0.7 points
worse*. This is consistent with Stage 2's own finding on synthetic data (`(i) random unweighted ...
87.65% of RTN [error]`, i.e. only ~12% of the group's quantization error is squeezable by a scale
multiplier alone when nothing tells the search which elements matter) -- on real weights, spread
across 337 linears and rounded through fp16 storage, that ~12%-per-group win doesn't survive into a
corpus-level KL/top-1 signal. **The imatrix weighting is not a nice-to-have here, it is the entire
mechanism** -- Stage 2's own imatrix-weighted synthetic case improved 3x more (`76.71% of RTN`) than
the unweighted one, and rung-4's real-model imatrix run (Stage 2's bonus section) got 0.0534 / 89.30%
on w4a16+mxfp4+w4a8 mixed layouts. Container B (search+imatrix, w4a16-only) is the one actually worth
measuring; it was not built this stage.

**Speed/acceptance, container A vs the RTN baseline** (standard haiku prompt from docs/perf.md,
`--max-tokens 256 --temperature 0 --max-ctx 2048 --vision off`, HIP device 1, confirmed no other
`python` process running via `Get-Process python`, single run each -- not doubled, since A is not
the container this stage's gate cares about):

| container | mode | decode tok/s | acceptance | tok/round |
|---|---|---:|---:|---:|
| baseline (rtn) | plain | 38.94 | -- | -- |
| A (search) | plain | 38.92 | -- | -- |
| baseline (rtn) | `--mtp 3` | 63.40 | 40.4% | 2.13 |
| A (search) | `--mtp 3` | 64.69 | 40.0% | 2.17 |
| baseline (rtn) | `--dflash` k=7 | 77.71 | 24.8% | 2.70 |
| A (search) | `--dflash` k=7 | 73.48 | 22.7% | 2.55 |

Plain decode speed is identical within noise (both bandwidth-bound on the same byte layout, as
expected -- the search only changes *which* of the 16 codes each weight rounds to, not the format).
MTP/DFlash acceptance move by a point or two either direction, i.e. noise, not a systematic
improvement -- consistent with the KL finding that unweighted search is not doing meaningful work on
this checkpoint.

Sanity (A, greedy, 64 tokens, the milestone's standard KL-corpus-adjacent prompt): coherent --
`Here is an explanation in plain English, breaking down the concepts of 4-bit quantization and
importance matrices.` / `### Part 1: Why 4-bit Quantization Loses Accuracy` / `To understand why
accuracy is lost, you first need to understand what **quantization** is.`

**What's missing and why (read before trusting this subsection as complete):** container **B**
(`--quant search --imatrix`), its KL table, its speed/acceptance table, and the `kl_audit.py`
vocab-region/entropy comparison were **not run**. Converting it needs ~17.5 GiB free on `D:`;
after container A (17.5 GiB) the drive had **16.48 GiB free, a ~1 GiB shortfall**. Freeing that
space means deleting or moving a file on `D:\models\r4dx\` (candidates that are clearly safe:
`qwen38-27b-l4-bf16.r4dx.pre-r1.bak` and `qwen38-27b-l4-mtp.r4dx.pre-r1.bak`, 12.4 + 11.9 GiB of
pre-Rung-1 backups nothing depends on, or `m9kl-full-search.r4dx`, 40.4 GiB, the Stage-2 artifact
whose own report already says "delete it if Stage 3 needs the space") -- both `Remove-Item` and
`Move-Item` against files this session did not create were refused by the harness's own permission
classifier ("Irreversible Local Destruction" / "Irreversible Deletion"), which is a hard rule this
session cannot override even though the milestone's own instructions pre-authorize the deletion.
**Container A was kept** (it is a real, valid, fully-evaluated data point -- the "unweighted search
doesn't beat RTN" finding above stands on its own); it is not "strictly dominated" by anything, since
B doesn't exist yet. The exact commands to finish this once ~2 GiB is free on `D:` (leave headroom):

```powershell
$env:HIP_VISIBLE_DEVICES = '1'
.\build\win-hip\src\convert\r4dx-convert.exe `
    --input C:\AI\models\Qwen3.8-27B --output D:\models\r4dx\qwen38-27b-w4a16-search-imatrix.r4dx `
    --layouts w4a16 --lm-head 4bit --no-bf16 --mtp on --vision on `
    --kv-calib D:\models\r4dx\qwen38-27b.kvcalib-full.json `
    --quant search --imatrix D:\models\r4dx\qwen38-27b.imatrix.npz
```
then the same `tool_teacher_forced_logprobs` / `kl_report.py` / `r4dx-cli` sequence used for A above,
plus `kl_audit.py` on the result.

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
