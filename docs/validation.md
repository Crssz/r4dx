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

> **2026-09-25:** a window of up to 10 rows is now row-identical to the step path, except where it
> straddles a change of the attention kernel's segment width (context 512, 1024, ... at `--max-ctx`
> >= 1024) -- [mtp.md](mtp.md), "Sampled rounds are bit-exact". The tool still has one path; this
> only removes the reason above for most positions. Its own rows (one `Prefill` of one token, then
> `DecodeStep`: every GEMM at M=1, attention at `q_len = 1`) are touched by one part of that fix:
> the decode attention now decides its lazy rescale per row, which moves the last bits of rows past
> 16 x segments keys. Re-run on `qwen38-27b-v6.r4dx` (`--max-ctx 4096`, all four dumps change):
> mean KL 0.03856 -> 0.03853, top-1 90.86% -> 91.03%, 2 positions above 1 nat either way; the two
> dumps differ from each other by mean KL 0.00039. The numbers in this file stand.

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
| thai_prose | 1023 | 0.15633 | 0.10060 | 0.83319 | 3.89099 | pos 206 (tok 53900 `'à¹Œ'`) | 78.20% | 98.53% | 15.610 | 16.991 | 8 |
| **ALL** | 4092 | **0.08794** | 0.04833 | 0.58740 | 3.89099 | thai_prose pos 206 | **87.00%** | 99.41% | 5.661 | 6.027 | 10 |

**Positions with KL > 1 nat** (10 total, all in code/Thai; none in `english_prose`/`python_source`):

- `cpp_source`: pos 526 (next token 874, `' no'`, KL 1.288), pos 662 (next token 9, `'*'`, KL 1.514).
- `thai_prose`: pos 206 (tok 53900 `'à¹Œ'`, KL 3.891), pos 257 (tok 38534 `'à¸•'`, KL 1.351), pos 258
  (tok 148410 `'à¹‰à¸­à¸‡'`, KL 2.035), pos 318 (tok 149334 `'à¸à¸²'`, KL 1.717), pos 636 (tok 157620
  `'à¹€à¸‰à¸¥'`, KL 2.376), pos 669 (tok 45596 `'à¸µà¹ˆ'`, KL 1.290), pos 949 (tok 35982 `'à¸§'`, KL 1.887), pos
  962 (tok 148947 `'à¸¹à¸'`, KL 1.808). Every Thai KL>1 token is a sub-syllable script fragment, not a
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

| streaming impl | hidden max\|diff\| | logits max\|diff\| | argmax agreement | per-position KL(controlâ€–streaming) |
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

| segment | ref mass in ids â‰¥148000 | KL from that region | KL per unit mass, low / high | mass-weighted \|Î” logp\|, low / high |
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

**Verdict.** Trustworthy as stated: the KL is `KL(P_bf16 â€– Q_w4a16)`, in nats, over the full
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

### Milestone 10: grid search + imatrix (2026-09-22, completed in the stage-4 review pass)

`r4dx-convert --quant search` chooses each `(row, 128-K group)`'s `(scale, zero)` by minimizing the
(optionally `--imatrix`-weighted) squared reconstruction error instead of taking the min/max grid;
`--quant rtn` is the historical round-to-nearest behaviour and remains the **default**. The on-disk
byte layout is identical in both modes (`docs/container-format.md`, "How the quantized values are
chosen") -- same tensor names, dtypes, shapes, sizes and offsets, verified below -- so this is a
pure "better bytes, same kernels" experiment.

Containers, all 64 layers, `--lm-head 4bit --no-bf16`, same full-forward KV calibration
(`qwen38-27b.kvcalib-full.json`), evaluated with `tool_teacher_forced_logprobs` + `kl_report.py`
against the bf16 reference dumps in `tools/reference/kl_out/ref`:

| id | container | `--quant` | `--imatrix` | layouts | convert wall |
|---|---|---|---|---|---|
| baseline | `qwen38-27b-w4a16-kvfull.r4dx` | `rtn` | -- | w4a16 | -- |
| A | `qwen38-27b-w4a16-search.r4dx` | `search` | -- | w4a16 | 250.996 s |
| B | `qwen38-27b-w4a16-search-imatrix.r4dx` | `search` | `qwen38-27b.imatrix.npz` | w4a16 | 275.809 s |
| v4 | `qwen38-27b-v4.r4dx` | `rtn` | -- | w4a16,w4a8,mxfp4 | -- |
| S | `m9kl-full-search.r4dx` | `search` | `qwen38-27b.imatrix.npz` | w4a16,w4a8,mxfp4 | -- |

#### Headline

| layout | `rtn` mean KL / top-1 | `search` alone | `search --imatrix` |
|---|---|---|---|
| w4a16 | 0.07241 / 88.40% | 0.07130 / 87.71% | **0.05342 / 89.30%** |
| w4a8 | 0.14340 / 82.72% | (not run) | **0.11641 / 84.78%** |
| mxfp4 | 0.09116 / 86.14% | (not run) | **0.07966 / 86.09%** |

**The importance weighting is the entire mechanism; the search on its own is worth nothing.**
Unweighted search lowers its own per-group objective on *every* group (0 regressions in 768 real
groups, see "Optimality audit") and still lands at 0.0713 / 87.71% -- mean KL within noise of `rtn`
and top-1 **0.7 points worse**. Add the imatrix and w4a16 mean KL drops 26% and top-1 gains 0.9
points; w4a8 (whose integer zero is pinned to 8, so the scale is its only free parameter) gains the
most top-1, +2.1 points; mxfp4 gains 12.6% mean KL at flat top-1. For scale: published llama.cpp
`Q4_K_M` on this checkpoint is 0.011-0.014 / 95-96%, so this closes roughly a quarter of the gap to
it without touching a kernel or a byte of layout.

#### Full KL table, container B (`kl_w4a16-search-imatrix.json`)

| segment | rows | mean KL | median KL | p99 KL | max KL | top-1 | top-5 | ppl ref | ppl test | KL>1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cpp_source | 1023 | 0.03427 | 0.01086 | 0.22265 | 1.00139 | 92.18% | 99.61% | 3.805 | 3.926 | 1 |
| english_prose | 1023 | 0.04028 | 0.02472 | 0.22622 | 0.73900 | 90.91% | 100.00% | 3.755 | 3.957 | 0 |
| python_source | 1023 | 0.04128 | 0.01902 | 0.29715 | 0.74457 | 91.89% | 99.80% | 4.605 | 4.774 | 0 |
| thai_prose | 1023 | 0.09787 | 0.06685 | 0.55205 | 2.01238 | 82.21% | 98.92% | 15.610 | 16.820 | 3 |
| **ALL** | 4092 | **0.05342** | 0.02841 | 0.35308 | 2.01238 | **89.30%** | 99.58% | 5.661 | 5.943 | 4 |

Every segment improves over both `rtn` and search-alone; the largest absolute gain is `thai_prose`
(0.1326 search-alone -> 0.0979), the segment `rtn` was worst on.

Container A's own table (`kl_w4a16-search.json`): cpp_source 0.04905 / 91.10%, english_prose
0.05096 / 89.44%, python_source 0.05261 / 90.13%, thai_prose 0.13259 / 80.16%, ALL 0.07130 / 87.71%.

#### Speed and acceptance -- the byte layout really is free

Standard haiku prompt, `--max-tokens 256 --temperature 0 --max-ctx 2048 --vision off`, HIP device 1,
nothing else on the GPU, every cell run twice (the two runs agree to <=0.1 tok/s):

| container | plain decode tok/s | `--mtp 3` tok/s (accept, tok/round) | `--dflash` k=7 tok/s (accept, tok/round) |
|---|---:|---|---|
| baseline (`rtn`) | 38.89, 38.93 | 63.51, 63.52 (40.4%, 2.13) | 77.72, 77.81 (24.8%, 2.70) |
| B (`search --imatrix`) | 38.92, 38.92 | **72.01, 72.07 (48.4%, 2.42)** | 77.09, 77.15 (24.5%, 2.68) |

Plain decode is identical (38.92 vs 38.89-38.93), which is the point: same bytes, same bandwidth,
same kernels. The `--mtp 3` row is **not** noise and should not be read as one -- MTP's draft head
lives inside the container and got better weights too, so its acceptance rises 40.4% -> 48.4% and
carries decode with it, **+13.5% tok/s for free**. `--dflash`'s drafter is a *separate*, unchanged
container (`qwen38-27b-dflash2-w4a16.r4dx`), so only the verifier changed and its acceptance barely
moves (24.8% -> 24.5%). Generation stays coherent (haiku + two-sentence explanation, both
containers).

#### What was verified about the mechanism, not just the numbers

- **Format invariance.** B vs the `rtn` baseline: 1514 tensors each, identical names, dtypes,
  shapes, byte sizes *and* byte offsets; only values differ. `git diff` over the milestone touches
  nothing under `src/model`, `src/kernels` or `third_party`.
- **Byte provenance.** The first 16-row tile of `text.layers.3.attn.o.w4a16.{wq,wsz}` was recomputed
  from the checkpoint with `tools/convert_ref/w4_ref.py` and matches, byte for byte, in all three
  containers with the matching reference mode (`rtn` for the baseline, unweighted search for A,
  `text.layers.3.attn.o`-weighted search for B). 22186/98304 codes differ between `rtn` and search
  and a further 34607 between search and search+imatrix, so the test is not vacuous: a container
  built with the wrong vector, or none, cannot pass its own row.
- **Byte-exactness through the real CLI.** `tools/convert_ref/selftest_compare.py` diffs
  `r4dx-convert --selftest` against the Python reference in all three modes (random fixture), and
  the same diff was repeated on a **real** tensor slice (16 x 6144 of `layers.3.self_attn.o_proj`
  with its real imatrix vector): all 7 tensors x 3 modes byte-exact.
- **Optimality audit.** On that real slice, in float64: `search` error <= `rtn` error on **768 of
  768** groups, both weightings (0 regressions), and on 20 randomly chosen groups an exhaustive
  re-evaluation of the whole documented candidate set (21 scales x 3 zeros, plus a least-squares
  refit of each candidate) found **no candidate that beats the one the converter chose**. Total
  weighted error: 89.15% of `rtn`'s unweighted, 50.36% of `rtn`'s under the real imatrix. Rounding
  the scale to the f16 the container actually stores does not break the guarantee either (0/768).
- **Imatrix plumbing.** The real run reports `imatrix coverage: 341 linear(s) weighted, 0 fell back
  to unweighted MSE`. A vector whose length does not match the linear's K is a hard error
  (`imatrix vector 'selftest' has length 128 but the linear's K is 6144`); a `savez_compressed`
  archive is a hard error naming the fix; a missing key warns per-linear and is counted in the
  coverage line, which says `WARNING` and goes to stderr when anything fell back; `--imatrix`
  without `--quant search`, and `--imatrix` with `--dflash-gguf` (the drafter shares none of the
  keys), are both rejected at argument-parse time.
- **No corpus contamination.** The imatrix corpus (`calib.txt` + `kv_calib_corpus/`) and the KL eval
  corpus (`kl_corpus/`) share exactly **one** line longer than 20 characters out of 467 --
  `from __future__ import annotations` -- and their longest common substring is 50 characters of
  Python import boilerplate. The imatrix win is not memorized eval text.
- **Reproducibility.** Container B was converted independently of Stage 2's all-layouts
  `m9kl-full-search.r4dx`, on a different drive, in a different session. Their `w4a16` logprob dumps
  are **byte-identical** (sha256 matches on `cpp_source` and `thai_prose`) and their overall mean KL
  agrees to all 16 digits (0.05342482327243944). Converter and inference path are both
  deterministic. The reference dumps in `tools/reference/kl_out/ref` were not touched (mtimes and
  sha256 unchanged from the rung-4 pass).

#### Why `rtn` is still the default

`--quant search` alone is measurably not an improvement (top-1 0.7 points worse) and costs ~2x the
conversion wall time, and `--imatrix` cannot be a default because it needs a capture file. Making
`search` the default would therefore have silently changed what a plain
`r4dx-convert --input ... --output ...` produces -- including the `--dflash-gguf` drafter path,
which has no imatrix at all and was never measured under search -- for no gain. The pair that *is*
worth using is explicit:

```powershell
$env:HIP_VISIBLE_DEVICES = '1'
.\build\win-hip\src\convert\r4dx-convert.exe `
    --input C:\AI\models\Qwen3.8-27B --output D:\models\r4dx\qwen38-27b-v5.r4dx `
    --layouts w4a16,w4a8,mxfp4 --lm-head 4bit --no-bf16 --mtp on --vision on `
    --kv-calib D:\models\r4dx\qwen38-27b.kvcalib-full.json `
    --quant search --imatrix D:\models\r4dx\qwen38-27b.imatrix.npz
```

#### Not done

- No all-layouts `--mtp on --vision on` production container was rebuilt with `search --imatrix`;
  the w4a8/mxfp4 numbers above come from Stage 2's `m9kl-full-search.r4dx`, which is `--mtp off
  --vision off` but otherwise identical in flags to `qwen38-27b-v4.r4dx` (neither MTP nor the vision
  tower participates in the `--vision off` teacher-forced forward, so the comparison is sound for
  KL; it does mean the w4a8/mxfp4 **acceptance** numbers were not measured).
- `kl_audit.py`'s vocab-region/entropy breakdown was not re-run for B.
- Container B lives on `C:\AI\r4dx-tmp\` rather than `D:\models\r4dx\`: `D:` had 16.48 GiB free and
  the container needs 17.5 GiB, and freeing space means deleting files this session did not create.
  Move it to `D:\models\r4dx\` when space allows; nothing in the tooling depends on its location.

### Milestone 11 / group size: what the w4a16 group buys (2026-09-22)

The w4a16 group -- how many contiguous `K` share one `(scale, zero)` pair -- is now a build option,
`R4DX_W4A16_GROUP` (default 128 as introduced here; **the "recipe" section below flipped the
default to 64** on the strength of what this section measures), reaching both the kernel
(`-DR4D_GEMM_W4_GROUP`) and the packer
(`kW4A16Group`) from one CMake cache variable; `docs/build-windows.md` "w4a16 group size" has the
mechanism and the three layers that stop the two sides drifting apart. The kernel packs 64
contiguous K per weight block and derives `bpg = group / 64`, so **64 is the only value below 128
it accepts unmodified** -- nothing under `third_party/libr4d/` was touched for this.

The whole trade is in one identity: a w4a16 weight costs `4 + 32/group` bits (4 nibble bits plus
one 32-bit `wsz` dword per 16 rows x group), so **4.25 bits at group 128, 4.50 at group 64** -- a
5.88% bigger weight stream for a finer quantization grid. `wq` does not change at all.

Both containers: all 64 layers, `--layouts w4a16 --lm-head 4bit --no-bf16 --mtp on --vision on
--quant search --imatrix qwen38-27b.imatrix.npz --kv-calib qwen38-27b.kvcalib-full.json`. The
group-64 one converted in 280.4 s and was deleted after measuring; the group-128 column is
`qwen38-27b-v5.r4dx`, the production container, whose w4a16 tensors are byte-identical in `wq` to
the group-64 one (same 12,980,060,160 bytes) and differ only in `wsz`. Both were measured in the
same session, HIP device 1, nothing else on the GPU.

| | group 128 (`v5`) | group 64 | delta |
|---|---:|---:|---|
| bits / weight | 4.2500 | 4.5000 | +5.88% |
| `w4a16.wq` bytes | 12,980,060,160 | 12,980,060,160 | 0 |
| `w4a16.wsz` bytes | 811,253,760 | 1,622,507,520 | x2 (+0.755 GiB) |
| w4a16 weights on disk | 12.845 GiB | 13.600 GiB | +0.755 GiB |
| `weights` in the VRAM breakdown | 15.5076 GiB | 16.2190 GiB | +0.711 GiB |
| mean KL vs bf16 | 0.05342 | **0.04214** | **-21.1%** |
| top-1 agreement | 89.30% | **90.71%** | **+1.41 pt** |
| top-5 containment | 99.61% | 99.73% | +0.12 pt |
| positions with KL > 1 | 4 | 2 | -2 |
| plain decode tok/s | 38.69, 38.66 | 36.39, 36.44 | **-5.8%** |
| `--mtp 3` tok/s (accept, tok/round) | 71.45, 71.24 (48.4%, 2.42) | 65.19, 65.20 (47.5%, 2.33) | -8.7% |
| `--dflash` k=7 tok/s (accept, tok/round) | 76.34, 76.62 (24.5%, 2.68) | 74.38, 74.68 (25.3%, 2.74) | -2.5% |

`tool_teacher_forced_logprobs` + `kl_report.py` against `tools/reference/kl_out/ref`, full 4-segment
corpus, 4092 rows (`kl_m11-v5-g128.json`, `kl_m11-g64.json` -- the group-128 run reproduces the
Milestone 10 headline 0.05342 / 89.30% to the last digit, which is the cross-check that the two
columns are the same measurement). Speed: `docs/perf.md`'s standard haiku prompt, `--layout w4a16
--vision off --think off --temperature 0 --max-tokens 256 --max-ctx 2048 --stats`, every cell twice
(the pairs agree to <=0.3%); both models stop at EOS (75-85 tokens) well before 256.

Per segment, group 64:

| segment | rows | mean KL | median KL | p99 KL | max KL | top-1 | top-5 | ppl ref | ppl test | KL>1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cpp_source | 1023 | 0.02901 | 0.00820 | 0.22107 | 0.39695 | 93.35% | 99.80% | 3.805 | 3.906 | 0 |
| english_prose | 1023 | 0.03186 | 0.01769 | 0.19746 | 0.30259 | 91.79% | 100.00% | 3.755 | 3.906 | 0 |
| python_source | 1023 | 0.03072 | 0.01465 | 0.21283 | 0.44035 | 93.74% | 99.90% | 4.605 | 4.749 | 0 |
| thai_prose | 1023 | 0.07698 | 0.05085 | 0.37236 | 2.18982 | 83.97% | 99.22% | 15.610 | 16.460 | 2 |
| **ALL** | 4092 | **0.04214** | 0.02241 | 0.28581 | 2.18982 | **90.71%** | 99.73% | 5.661 | 5.876 | 2 |

Every segment improves, and the two hardest ones improve most in the places that matter: `cpp_source`
loses its only KL>1 position outright and `thai_prose`'s p99 drops 33% (0.552 -> 0.372) even though
its single worst position gets slightly worse (2.012 -> 2.190). Group 64 is not a different kind of
fix from the imatrix -- it is the same fix applied twice as often.

#### What decode pays

**Plain decode is bandwidth, and this is what bandwidth looks like: +5.88% bytes, -5.8% tok/s.**
38.69 -> 36.42 tok/s is the weight stream and nothing else -- same kernel, same launch geometry,
same activation path; `wq` is bit-identical and only `wsz` doubled. The two speculative modes are
not a second independent measurement of the same thing: `--mtp 3` loses more (-8.7%) because its
verify step re-reads the same larger weights *and* its own draft head got slightly less accurate in
this run (48.4% -> 47.5% acceptance), while `--dflash` loses least (-2.5%) because most of its wall
time is the separate, small drafter -- whose own acceptance actually rises (24.5% -> 25.3%, 2.68 ->
2.74 tok/round) since a group-64 drafter had to be converted too.

That last point is a cost in its own right: **the group is a property of the whole fleet, not of one
container.** A group-64 build refuses every group-128 container at `Container::Load`, drafters
included, so switching means re-converting the production container, the DFlash2 drafter and the
fixed test containers together. Both directions of that refusal were exercised by hand (a group-64
`r4dx-cli` on `v5`, and the default build on the group-64 container) and both print the group the
container carries, the group the kernel reads, and the `-DR4DX_W4A16_GROUP=` to fix it.

#### Gates

- `ctest --preset win-hip` (group 128, the default): **64/64 passed**, 783.40 s, 1 skipped
  (`test_kernel_bandwidth`, as always). Re-run at the end of the milestone it is 62 passed / 2
  **Not Run** / 1 skipped, 584.72 s: `reference_manifest` and `reference_dflash2` launch the
  reference venv's `python.exe`, and that venv was deleted from the machine partway through this
  session by something outside this work (`Could not find executable
  .../.venv-rocm10/Scripts/python.exe`). Nothing in those two tests is touched by this milestone
  and both passed on this exact tree earlier in the session, in the 64/64 run above.
- `ctest --preset win-hip-g64`: the same suite, **62 passed / 2 Not Run** (the same two venv tests)
  **/ 2 skipped** (`test_kernel_bandwidth`, plus `test_dflash_e2e` -- see "Not done"), 424.53 s,
  with `R4DX_TEST_CONTAINER_DIR` pointed at group-64 copies of the fixed test containers
  (`tests/model/test_container_path.h` -- without the override a non-default-group build cannot run
  these tests at all, because it correctly refuses the group-128 originals). Every w4a16 test that
  runs on the default build runs here too, at group 64: `test_gdn_layer`, `test_final_lm_head`,
  `test_forward_smoke`, `test_mtp`, `test_attn_layer`, `test_teacher_forced_logprobs`,
  `test_dflash_*`.
- `test_attn_layer` is worth singling out, because it failed first and the failure was real:
  it reads the container through a raw `SafetensorsReader`, **not** `Container::Load`, so it gets no
  group check -- and its `-D`-supplied container path was overriding the in-file default, so the
  group-64 binary was reading group-128 `wsz` bytes. The w4a16 pass reported
  `prefill norm rel err=-nan(ind)`. That is the guard's whole thesis, demonstrated by accident: the
  three paths that DO go through `Container::Load` refused the same container with a clear message,
  and the one path that bypassed it produced garbage. Fixed by routing that macro through
  `ContainerPath` as well.
- `tests/convert` is group-agnostic on **both** builds: every int4 quantizer, packer, search and
  kernel-literal decode runs at 64 *and* 128 whatever `R4DX_W4A16_GROUP` the build is, against
  `tests/convert/fixtures/*_g64.bin` (the group-128 fixtures keep their historical unsuffixed names
  and are byte-unchanged).
- `tools/convert_ref/selftest_compare.py` (rung 1, end to end through the real CLI) passes against
  both exes and reads the group out of the container the exe just wrote: `w4a16=128 w4a8=128` for
  the default build, `w4a16=64 w4a8=128` for the group-64 build -- which is also the check that
  splitting `kInt4Group` into `kW4A16Group`/`kW4A8Group` actually took effect.

#### One assertion had to be corrected

`tests/convert/test_quant_search.cpp` property (ii) asserted that the *unweighted* search beats RTN
under the *imatrix-weighted* metric. Nothing guarantees that -- the unweighted search does not
optimize that objective -- and it held at group 128 only by a 0.27% margin. Running the same case at
group 64, where the outlier group is half as wide and the outlier therefore twice as dominant, flips
it to 0.11% the wrong way. The assertion now compares the unweighted search against RTN under the
unweighted metric, where the guarantee **is** structural (the RTN grid is candidate 0 and later
candidates must win strictly); the weighted numbers are still printed for comparison. The three
claims the case actually exists to make -- imatrix beats RTN, by a wide margin, and beats the
unweighted search -- were unchanged and pass at both groups.

#### Not done

- Only the w4a16 group was made configurable. w4a8's group stays pinned at 128 and mxfp4's at 32;
  splitting the shared `kInt4Group` constant was what made w4a16's independent, so doing the same
  for w4a8 is now a one-line change plus a second cache variable, if a reason ever appears.
- `test_dflash_e2e` was not run on the group-64 build: it reads `qwen38-27b-v3.r4dx`, a 42 GiB
  production container, and a group-64 copy of it is not worth 42 GiB of disk for one test.
- Group 32 (4.0 + 1.0 = 5.0 bits/weight) was not measured: the kernel's `bpg = group / 64` makes it
  a libr4d change, which this milestone excluded.

### Milestone 11 / sensitivity: where the remaining nats live (2026-09-22)

The group-size experiment above answered "does a finer grid help?" (yes, a little, at a price the
decode budget cannot afford). It did not answer the question underneath it: **which weights is the
0.0534 actually coming from?** 4-bit error is not spread evenly over 25.9 G parameters, and if it
is concentrated somewhere cheap, that is the place to spend the next bit.

#### The instrument

`r4dx-convert --keep-bf16 <regex>` (this milestone; `src/convert/include/r4dx_convert/keep_bf16.hpp`,
`docs/container-format.md` "Quantized layout tensors") writes every linear whose container base name
matches as `<base>.bf16.w` and nothing else. `Container::Load`'s `LoadQuantLinearWithFallback` --
now on **every** quantized body linear, not just the three R1 tensors -- sees a base with no
quantized form and falls that one linear back to bf16, leaving the rest of the container in the
requested `w4a16`. So one conversion per tensor class gives a model that is 4-bit everywhere except
the class under test, and the KL that disappears against the all-4-bit baseline is that class's
share of the error. `Container::Load` prints the fallback count on stderr, which is each run's own
check that the regex selected what it meant (`r4dx: 48 linear(s) ... loaded as bf16`).

Method, identical for every row: the Milestone 10 recipe
(`--layouts w4a16 --lm-head 4bit --no-bf16 --mtp on --vision on --quant search --imatrix
qwen38-27b.imatrix.npz --kv-calib qwen38-27b.kvcalib-full.json`) plus one `--keep-bf16` regex;
`tool_teacher_forced_logprobs --layout w4a16 --max-ctx 4096 --vision off` over
`tools/reference/kl_corpus`; `kl_report.py` against the bf16 reference dump. Conversions ran
240-272 s each, KL dumps 144-210 s each; every container was deleted immediately after its dump.
The baseline was re-measured in the same session with the same binary and reproduced the Milestone
10 headline exactly: **mean KL 0.053425, top-1 89.296%**, `weights=15.5076 GiB`.

`+GiB` is the exact weight-byte delta the converter reports (bf16 minus the `w4a16` form it
replaced), which is what the GPU actually has to stream; the measured `weights=` line is quoted
next to it and agrees to within the driver's allocation granularity. For `lm_head` the two differ
by construction: the recipe's `--lm-head 4bit` writes three quantized layouts on disk, only one of
which is ever loaded, so the on-disk delta (0.481 GiB) understates the runtime delta (1.739 GiB) --
the table uses the runtime one, and the measured VRAM line confirms it.

#### The ranked table

Baseline `w4a16` = 0.053425 / 89.296% / thai 0.097875 / 82.209%. "nats" = mean KL in nats/token.

| # | class kept in bf16 | linears | +GiB | measured `weights=` | mean KL | nats recovered | **nats / GiB** | top-1 | thai KL | thai top-1 | KL>1 |
|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | `attn.k` + `attn.v` | 32 | 0.2295 | 15.6951 | 0.04956 | 0.00386 | **0.01683** | 89.91% | 0.09398 | 82.60% | 1 |
| 2 | `attn.o` | 16 | 0.6885 | 16.2107 | 0.05094 | 0.00248 | **0.00361** | 89.39% | 0.09315 | 82.21% | 3 |
| 3 | `lm_head` | 1 | 1.7391 | 17.2466 | 0.04762 | 0.00580 | **0.00334** | 90.42% | 0.08950 | 83.38% | 3 |
| 4 | `gdn.out_proj` | 48 | 2.0654 | 17.6169 | 0.04843 | 0.00499 | 0.00242 | 89.76% | 0.08565 | 82.99% | 1 |
| 5 | *last 8 layers, all linears* | 42 | 4.1595 | 19.6614 | 0.04346 | 0.00997 | 0.00240 | 90.27% | 0.07406 | 84.46% | 4 |
| 6 | `gdn.in_proj_qkv` | 48 | 3.4424 | 18.9607 | 0.04627 | 0.00715 | 0.00208 | 90.03% | 0.08468 | 83.97% | 1 |
| 7 | *first 8 layers, all linears* | 42 | 4.1595 | 19.6614 | 0.04733 | 0.00610 | 0.00147 | 89.81% | 0.07628 | 84.75% | 6 |
| 8 | `gdn.in_proj_z` | 48 | 2.0654 | 17.6169 | 0.05059 | 0.00284 | 0.00137 | 89.35% | 0.09255 | 82.50% | 5 |
| 9 | `mlp.gate_up`, layers 32-63 | 32 | 7.8027 | 23.3103 | 0.04335 | 0.01008 | 0.00129 | 90.05% | 0.07912 | 83.77% | 1 |
| 10 | `mlp.down` | 64 | 7.8027 | 23.2263 | 0.04550 | 0.00792 | 0.00102 | 90.30% | 0.08188 | 83.48% | 2 |
| 11 | `mlp.gate_up`, all 64 layers *(sum of the two halves, not measured directly)* | 64 | 15.6055 | -- | -- | 0.01334 | 0.00086 | -- | -- | -- | -- |
| 12 | `attn.qg` | 16 | 1.3770 | 16.9138 | 0.05259 | 0.00083 | 0.00060 | 89.44% | 0.09715 | 82.40% | 2 |
| 13 | `mlp.gate_up`, layers 0-31 | 32 | 7.8027 | 23.3103 | 0.05017 | 0.00326 | 0.00042 | 89.42% | 0.09339 | 82.50% | 4 |

`mlp.gate_up` is the one class that could not be measured whole: 11.4 G parameters in bf16 is
+15.6 GiB, which puts `weights` at 31.1 GiB on a 31.86 GiB card and leaves nothing for the KV cache
-- WDDM would not fail the allocation, it would page it over PCIe and quietly invalidate the run.
It was measured as two halves instead, and row 11 is their **sum**, marked as such.

For scale: the group-64 arm from the previous section recovers 0.01128 nats for +0.7114 GiB, i.e.
**0.01586 nats/GiB**. Only one tensor class in this whole table beats a plain finer grid, and it
beats it by 6%.

#### The cheap wins, and the expensive ones

**`attn.k`/`attn.v` are the anomaly, by an order of magnitude.** 0.0169 nats/GiB is 4.7x the next
class and 40x the worst. These are the two smallest quantized linears in the model -- 16 layers x
[1024, 5120], 0.168 G parameters, 0.65% of the quantized weights; un-quantizing both costs 0.2295
GiB, 1.5% of the 15.5 GiB weight stream -- and they carry **7.2%** of the error. Keeping them in
bf16 also cuts the corpus's KL>1 positions from 4 to 1. The reason is structural and specific: `attn.k`/`attn.v`'s outputs are not consumed once and
discarded like every other projection's, they are *written into the fp8 KV cache and re-read by
every later position in the sequence*. A weight error there is the only one in the model that
compounds along the context rather than along depth. (`docs/r9700.md` R1 moved these two out of
forced-bf16 into the quantized family for a 0.336 GB/token bandwidth win; this is the bill for that,
and it is small in bytes and large in nats.)

**`lm_head` and `attn.o` are the other two cheap wins**, at 0.0033 and 0.0036. `lm_head` matches the
Rung 4 follow-up's independently measured "lm_head share ~0.009" in direction (that experiment
compared against a different baseline); `attn.o`'s per-byte cost is high for the same reason
`attn.qg`'s is low -- see below.

**The MLP is where the bytes are and where the nats are not.** `mlp.gate_up` + `mlp.down` is 17.1 G
of 25.9 G parameters -- 66% of the weight stream -- and buys 0.021 of the 0.053, at the two worst
per-byte rates in the table. If you have one gigabyte to spend, spending it on the MLP is the single
worst thing you can do with it.

**`attn.qg` is the cheapest class to quantize and the most expensive to un-quantize** (0.0006
nats/GiB: 6x worse per byte than its own layer's `attn.o`, 28x worse than its own layer's
`attn.k`/`v`). It is the fused query+gate matrix: half of its output is a *gate* that goes through a sigmoid, which
is contractive, and the query half is immediately RMS-normed per head, which removes exactly the
kind of scale error 4-bit quantization introduces. Leave it at 4 bits.

**Depth matters as much as tensor class.** The two 8-layer bands are the same 42 linears and the
same 4.1595 GiB, and the last 8 recover 63% more than the first 8 (0.00997 vs 0.00610). The split is
sharper inside a single class: `mlp.gate_up` in layers 32-63 recovers **3.1x** what the identical
bytes recover in layers 0-31 (0.01008 vs 0.00326). Any future mixed-precision scheme should be
depth-aware, not just class-aware. (Both bands, notably, help Thai far more than the average --
thai KL 0.0763/0.0741 against 0.0979 baseline -- which is consistent with the multilingual tail
being hurt by accumulated per-layer error rather than by one bad tensor class.)

**The classes are close to additive.** The nine disjoint classes' recoveries sum to 0.04923 against
a baseline of 0.05342 -- **92.1%** of the total. The nine cover every quantized `text.layers.*`
linear plus `lm_head`; the only quantized weights no regex here touched are the `mtp.*` head's, so
the missing 7.9% is that plus genuine cross-class interaction. Close enough that the table can be
read as a budget and small combinations estimated by addition without much guilt.

#### What an 8-bit weight kernel would buy -- ESTIMATE, not measured

libr4d has no 8-bit-weight GEMM, so this is arithmetic, not a measurement. Stated so it can be
checked if anyone builds one.

Two assumptions, both first-order:

1. **Bytes.** A `w8a16` weight at group 128 costs `8 + 32/128 = 8.25` bits against w4a16's 4.25 and
   bf16's 16, so it pays **(8.25-4.25)/(16-4.25) = 34.0%** of the bf16 extra bytes measured above.
2. **Error.** Uniform quantization MSE falls 4x per added bit, so int8's weight-reconstruction MSE
   is `1/256` of int4's, and KL is quadratic in a small weight perturbation (KL ~ ½ δᵀFδ), hence
   proportional to that MSE. `w8a16` should therefore recover **99.6%** of what bf16 recovers.

Together: **~2.93x the nats per GiB of the bf16 column**, for every class.

| class | `w8a16` +GiB (est) | nats recovered (est) | nats/GiB (est) |
|---|--:|--:|--:|
| `attn.k` + `attn.v` | 0.078 | 0.00385 | 0.0492 |
| `attn.o` | 0.234 | 0.00247 | 0.0106 |
| `lm_head` | 0.592 | 0.00578 | 0.0098 |
| `gdn.out_proj` | 0.703 | 0.00498 | 0.0071 |
| `gdn.in_proj_qkv` | 1.172 | 0.00713 | 0.0061 |

The interesting line is the combination. Putting the top three cheap classes
(`attn.k`/`v` + `attn.o` + `lm_head`) at 8 bits and leaving the other 92% of the weights at 4 bits
costs an estimated **+0.90 GiB** (15.51 -> 16.41 GiB of weights, +5.8%) and should land mean KL near
**0.0413**, versus 0.0421 for the group-64 arm at +0.71 GiB. That is roughly the same purchase as
group 64 -- which is to say: **still not enough.** llama.cpp `Q4_K_M` reaches 0.011-0.014 on this
same checkpoint at 4.5 bits/weight. Neither a finer grid nor selective 8-bit closes a gap that a
better *scheme* closes entirely inside the same budget, and the per-class table says why: the error
is not hiding in one class waiting to be paid off. Nine classes contribute between 1.6% and 25% of
it each, and the
two that contribute most (`mlp.gate_up`, `mlp.down`) are exactly the two nothing can afford to widen.

#### What this says to do next

- **Cheap and worth doing regardless of scheme:** `attn.k`/`attn.v` at higher precision. +0.23 GiB
  (1.5% of the weight stream, so ~1.5% of decode by the group-size section's bytes-to-tok/s
  relation) removes 7.2% of the KL and 3 of the corpus's 4 KL>1 positions. Today's `--keep-bf16 "^text\.layers\.\d+\.attn\.[kv]$"` already builds that container; it needs no
  new kernel.
- **Depth-aware allocation** before class-aware allocation. A scheme that spends its extra bits on
  layers 32-63 gets 3x the return of one that spreads them evenly.
- **Do not** spend bytes on `attn.qg` or on the first half of `mlp.gate_up`.
- The real target remains a `Q4_K_M`-class scheme (super-block scales, a second-level scale
  quantization) rather than more bits on the current one. This table's job was to find out whether
  there was a shortcut. There is a small one (`attn.k`/`v`) and no large one.

#### Gates

```
ctest --preset win-hip     97% tests passed, 2 tests failed out of 66
                           Total Test time (real) = 739.43 sec
                           FAILED: 2 - reference_manifest (Not Run), 3 - reference_dflash2 (Not Run)
ctest --preset win-hip -R keep_bf16
                           1/2 Test #10: convert_keep_bf16 .......   Passed    0.01 sec
                           2/2 Test #49: test_keep_bf16 ..........   Passed   25.82 sec
                           100% tests passed, 0 tests failed out of 2
```

`reference_manifest` and `reference_dflash2` are the two pre-existing failures from the previous
section: they launch the reference venv's `python.exe`, and that directory is gone from this machine.
Everything else, including both new tests, passes.

#### Not done

- `mlp.gate_up` as one class (see above -- it does not fit in 32 GiB). Row 11 is the sum of two
  halves, which the 92.1% additivity result makes credible but does not prove.
- Nothing was measured in *combination*: every row is one class against the same baseline. The
  additivity check is the only evidence that combinations add, and it is an aggregate one.
- No decode-throughput numbers. Every row's cost is in bytes; the group-size section already
  established that plain decode on this card tracks weight bytes almost exactly (+5.88% bytes ->
  -5.8% tok/s), so tok/s for these rows is predictable from `+GiB` and was not worth 13 more GPU
  runs.
- The `w8a16` table is arithmetic from two stated assumptions, not a measurement, and there is no
  8-bit-weight kernel in libr4d to measure.

### Milestone 11 / recipe: what v6 spends its extra gigabyte on (2026-09-22)

The two sections above are the two halves of one question. "Group size" measured a lever that makes
*every* weight slightly more accurate; "sensitivity" measured which weights are worth making a lot
more accurate. This section spends a fixed budget -- **at most +1.5 GiB of weights over
`qwen38-27b-v5.r4dx`** -- using both tables, and ships the result as
`D:\models\r4dx\qwen38-27b-v6.r4dx`.

#### The decision, and the arithmetic it was made on (before anything was converted)

Everything on the menu, ranked by the only currency that matters here -- nats of mean KL recovered
per GiB added to the weight stream, since decode on this card is weight-bandwidth-bound and a GiB
costs the same ~6% of tok/s wherever it is spent:

| lever | +GiB | nats recovered | **nats/GiB** |
|---|--:|--:|--:|
| `--keep-bf16` `attn.k`+`attn.v` | 0.2295 | 0.00386 | **0.01683** |
| w4a16 group 64 | 0.7114 | 0.01128 | **0.01586** |
| `--keep-bf16` `attn.o` | 0.6885 | 0.00248 | 0.00361 |
| bf16 `lm_head` | 1.7391 | 0.00580 | 0.00334 |
| everything else (9 more rows) | -- | -- | <= 0.00242 |

There is a cliff after the second row: the third-best buy is **4.4x worse per byte** than the
second. So the budget was filled greedily and then deliberately left short:

- **group 64** -- +0.7114 GiB, 0.01128 nats.
- **`--keep-bf16 "^text\.layers\.[0-9]+\.attn\.[kv]$"`** -- priced at group-64 rates, not the
  group-128 rates in the table above: the 32 k/v tensors are 167,772,160 parameters, so bf16 is
  0.3125 GiB against 0.0879 GiB at 4.5 bits/weight, i.e. **+0.2246 GiB**, not +0.2295. Their nats
  were discounted the same way: group 64 already recovered 21.1% of the error in every class, so
  k/v's remaining share is `0.00386 x (1 - 0.211) = 0.00305` nats, not 0.00386.
- **Predicted total: +0.936 GiB -> 16.444 GiB of weights; mean KL 0.05342 - 0.01433 = 0.0391;**
  decode `38.69 x (1 - 0.0604 x 1.30) ~ 35.7-36.2 tok/s` (the 1.30 is the measured
  decode-loss-per-weight-byte amplification from the group-size section: +4.59% weights cost
  -5.95% tok/s).

**What was deliberately *not* bought, with 0.564 GiB of budget still on the table:** `attn.o` is
+0.674 GiB at group-64 rates, which overruns the budget (1.610 GiB total) *and* buys 0.00196 nats
after the same 21.1% discount -- 0.0029 nats/GiB, one fifth the rate of what was bought. Spending
the rest of the budget at a fifth of the rate is not a better container, it is a slower one. The
budget is a ceiling, not a target.

**`lm_head` stays 4-bit.** bf16 is +1.7391 GiB on its own -- over budget by itself -- at 0.00334
nats/GiB, and there is no cheaper setting: `--lm-head` takes *layout* tokens (`4bit`, `bf16`,
`mxfp4`...), and no 8-bit weight kernel exists to offer an 8.25-bit middle.

#### Predicted vs measured

| | predicted | measured | error |
|---|--:|--:|--:|
| weights in VRAM | 16.444 GiB | **16.4065 GiB** | -0.23% |
| mean KL | 0.0391 | **0.038507** | -1.5% |
| plain decode | 35.7-36.2 tok/s | **35.96 / 35.86** | in range |

**Read the weights row carefully** (adversarial-review note). `weights=` is a free-VRAM delta from
`hipMemGetInfo` around `Container::Load` (`src/model/model.cpp`), not a byte count, so it carries
the driver's per-allocation rounding -- and v6 makes 32 fewer device allocations than v5, because a
kept linear is one bf16 buffer where a quantized one is `wq` + `wsz`. Summed straight off the two
containers' headers, what a `--layout w4a16` load actually uploads grows by **+0.9802 GiB**
(`wsz` 0.7555 -> 1.5013, `wq` 12.0886 -> 12.0105, plus 0.3125 of bf16 `attn.k`/`v`), against the
+0.936 predicted and the +0.8989 the VRAM line reports. So the error on the *increment* is +4.7% in
bytes and -4.0% as VRAM measures it; the -0.23% above is that same error divided by the 16.4 GiB
total rather than by the ~0.9 GiB actually being predicted. Most of the byte-side gap is inherited:
the group-size section's `+0.7114 GiB` was itself a `weights=` delta, where the header arithmetic
for the same change is +0.7556 GiB.

With that caveat stated, the per-class sensitivity table predicts the composite container to better
than 2% on the totals it was asked about. That is the useful result of this milestone independent
of v6 itself: the
classes are close enough to additive, and group size close enough to a uniform multiplier, that
recipes can be *designed* on paper from the two tables instead of converted and measured one by
one at ~7 minutes of CPU and ~2.5 minutes of GPU apiece.

#### v6, measured

`D:\models\r4dx\qwen38-27b-v6.r4dx`, 42.74 GiB on disk, 417.9 s to convert:

```powershell
$env:HIP_VISIBLE_DEVICES = '1'
.\build\win-hip\src\convert\r4dx-convert.exe `
    --input C:\AI\models\Qwen3.8-27B --output D:\models\r4dx\qwen38-27b-v6.r4dx `
    --layouts w4a16,w4a8,mxfp4 --lm-head 4bit --no-bf16 --mtp on --vision on `
    --kv-calib D:\models\r4dx\qwen38-27b.kvcalib-full.json `
    --quant search --imatrix D:\models\r4dx\qwen38-27b.imatrix.npz `
    --keep-bf16 "^text\.layers\.[0-9]+\.attn\.[kv]$"
```

KL against the bf16 reference on `tools/reference/kl_corpus` (4092 rows, 4 segments), all three
layouts out of the same file. v5 columns are this session's re-measurement of the same corpus, not
transcribed numbers:

| layout | mean KL v5 | **mean KL v6** | top-1 v5 | **top-1 v6** | thai KL v5 | **thai v6** | KL>1 | `weights=` |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| w4a16 | 0.05342 | **0.03851** | 89.30% | **90.93%** | 0.09787 | **0.07409** | 4 -> **2** | 16.4065 GiB |
| w4a8 | 0.11641 | **0.10405** | -- | **85.24%** | -- | 0.20799 | 16 | 15.6951 GiB |
| mxfp4 | 0.07966 | **0.07764** | -- | **86.39%** | -- | 0.15016 | 11 | 15.6951 GiB |

w4a16 is **-27.9% mean KL and +1.64 points of top-1** for +5.80% of weight bytes. w4a8 and mxfp4
improve too -- **-10.6%** and **-2.5%** -- although neither one's group changed and neither one was
the target: `--keep-bf16` drops a linear from *every* layout at once, so the bf16 `attn.k`/`attn.v`
is in all three containers. The 4x spread between w4a8's gain and mxfp4's is not explained here;
the plausible reading is that w4a8 quantizes activations as well, so the K and V it writes into the
fp8 cache start from a worse place and have more to gain from exact projections, but that is a
hypothesis, not something this measurement isolates. Per-segment detail is in
`tools/reference/kl_out/kl_v6-{w4a16,w4a8,mxfp4}.json`.

Decode, `docs/perf.md`'s standard prompt (`--layout w4a16 --vision off --think off --temperature 0
--max-tokens 256 --max-ctx 2048 --stats`), HIP device 1 with the GPU to itself, twice each:

| path | v5 | **v6** | delta | acceptance / tok-round |
|---|--:|--:|--:|---|
| plain | 38.69, 38.66 | **35.96, 35.86** | **-7.2%** | -- |
| `--mtp 3` | 71.45, 71.24 | **65.93, 66.11** | **-7.5%** | 47.6% / 2.40 (was 48.4% / 2.42) |
| `--dflash` k=7 | 76.34, 76.62 | **72.93, 72.71** | **-4.8%** | 24.9% / 2.71 (was 24.5% / 2.68) |

Prefill is unchanged (580-641 tok/s on a 29-token prompt; prefill is compute-bound, not
weight-bound). VRAM: `weights=16.4065 GiB`, 17.01 GiB total resident plain, 17.43 GiB with
`--mtp 3`, 19.04 GiB with the DFlash2 drafter. A 64-token greedy generation on a fresh prompt
("Explain in plain English why quantizing a neural network to 4 bits usually costs accuracy...")
comes back as ordinary well-formed English prose with correct Markdown structure -- no repetition,
no token garbage, 35.81 tok/s.

**So the bargain is: -27.9% KL, -7.2% plain decode, -4.8% on the fastest path.** Plain decode lost
1.23x its weight-byte increase (+5.80%), close to the 1.30x the group-size experiment measured on a
pure group change, so the amplification is a property of the weight stream and not of this
particular recipe. DFlash2 loses less than the bytes alone would predict, because a speculative
round amortizes one pass over the weight stream across 2.4-2.7 accepted tokens.

#### The drafter has to be re-converted too, and *how* matters

A group-64 build refuses the group-128 DFlash2 drafter (`DflashDraftWeights::Open` runs the same
`CheckW4a16Group` guard), so `D:\models\r4dx\qwen38-27b-dflash2-w4a16-g64.r4dx` was converted from
the same `Qwen3.8-27B-DFlash2-Q8_0.gguf`. The first attempt used the converter's default `--quant
rtn` and cost **6.4 tok/s**:

| drafter | dflash k=7 decode | acceptance | tok/round |
|---|--:|--:|--:|
| `--quant rtn` (the default) | 66.27, 66.32 | 21.4% | 2.47 |
| `--quant search` | **72.93, 72.71** | **24.9%** | **2.71** |

Both measured against the same v6 target in the same session. This is worth recording because it
contradicts the shape of Milestone 10's finding for the *main* model, where `--quant search`
without an imatrix was worth nothing: on the drafter it is worth 9.6% of the DFlash2 decode rate.
The mechanism is presumably that the drafter's job is agreement with a target rather than accuracy
in its own right, and acceptance is a much sharper function of small logit errors than KL is. The
shipped drafter is the `search` one. (`--imatrix` remains rejected on the `--dflash-gguf` path --
the drafter shares none of the main model's keys.)

#### What this breaks, and the way out

`R4DX_W4A16_GROUP` now defaults to **64**, so `build/win-hip` refuses every container packed
before v6 -- `v5`, `v4`, `v3`, the 4-layer test containers, the `w4a16` drafter -- by name, with
both numbers, and with the fix in the message, *whenever the run selects the `w4a16` layout*.
(Adversarial-review fix: the guard was originally unconditional, which also refused
`--layout mxfp4` / `--layout w4a8` on those same containers and refused the bf16 and mxfp4
drafters, none of which read a `.w4a16.wsz` byte. Those load normally now -- `docs/build-windows.md`
"w4a16 group size" layer 2.) This was a deliberate trade: v6 is only the
production container if the production build reads it without a flag. The escape hatch is the
`win-hip-g128` preset (verified this session: it builds clean and loads `qwen38-27b-v5.r4dx` at
`weights=15.5076 GiB`, decoding normally), and `docs/build-windows.md` "w4a16 group size" has the
whole story including the trap that an *existing* build directory keeps its cached 128.

`tests/model` reads 4-layer containers at hard-coded group-128 paths, so group-matched copies were
converted once into `D:\models\r4dx\g64\` and `R4DX_TEST_CONTAINER_DIR` points the suite at them.
`qwen38-27b-l4-mtp-draftvocab.r4dx` was regenerated there with a fresh arbitrary 4096-id subset
(the 1416 distinct ids in `kl_corpus/tokens.json`, padded from 0 -- the reduced-vocab test is about
correctness plumbing, not coverage), so the reduced-vocab draft head keeps its automated coverage
on the default build rather than silently dropping to `[SKIP]`.

#### What a `w8a16` kernel milestone should target

The sensitivity section's arithmetic (marked there as arithmetic, not measurement) says an 8-bit
weight layout would recover ~99.6% of a class's nats at 34.0% of bf16's extra bytes -- ~2.93x the
nats/GiB of every row in the table. Against v6 rather than v5, the concrete target is:

- v6 already pays bf16 for its k/v bytes, which is 2x more than it needs to. 167,772,160
  parameters at 8.25 bits is 0.1611 GiB against 0.0879 at 4.5, so 8-bit k/v is **+0.0732 GiB**
  instead of +0.2246 -- it **gives back 0.1514 GiB** for 99.6% of the same nats.
- `attn.o` (503,316,480 parameters) at 8 bits is **+0.2197 GiB** for ~0.00196 nats = **0.0089
  nats/GiB**, which finally clears the bar that kept it out of v6. `lm_head` (1,271,398,400) is
  **+0.5550 GiB** for ~0.00458 = **0.0083**. Both are group-64-discounted, as above.
- All three together: `+0.8989 - 0.1514 + 0.2197 + 0.5550 = **+1.52 GiB** over v5 -- right at the
  ceiling this milestone was given -- for `0.03851 - 0.00196 - 0.00458 = **KL ~0.0320**`.

That is the honest ceiling of this direction, and it is the reason to state the target now: ~0.0320
at a comparable bit budget is still **2.3-2.9x** llama.cpp `Q4_K_M`'s 0.011-0.014 at 4.5 bits on
this same checkpoint. Nine classes each carry 1.6-25% of the error and the two largest are the two nothing
can afford to widen, so no amount of per-class promotion closes that gap. A `w8a16` kernel is worth
building for the ~17% it buys and for unblocking mixed-precision recipes generally -- it is not
worth building in the belief that it closes the gap to `Q4_K_M`. Closing that gap needs a better
4-bit *scheme* (llama.cpp's k-quants spend their budget on a second-level quantization of the
scales themselves, which is a different shape of idea from anything measured in Milestone 11), and
that is a kernel milestone of its own.

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
