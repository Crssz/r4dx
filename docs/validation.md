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

Not yet built. Run the *whole* model (both `transformers`, CPU-offloaded a layer at a time given
the 27B size, and r4dx end to end on HIP device 1) on a handful of real prompt tokens and diff
final logits. This is where rung 3's per-layer tolerances compound -- expect a visibly looser
bound than any single layer's, and a plan for *how much* looser (e.g. `rel_err` growing roughly
with `sqrt(num_layers)` for independent per-layer error, `2e-2 * sqrt(64) ~ 0.16` as a first guess)
needs to be pinned down empirically once `src/model`'s layer graph exists, not guessed here.

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
