# tools/reference

Python reference/validation tooling for r4dx. These scripts don't touch the C++ engine at all --
they run the *real* HF `transformers` implementation of Qwen3_5 (read-only, from the reference
venv) and dump its numbers to disk, so `src/model`'s tests have a ground truth to diff against.
See `docs/validation.md` for where this fits in the overall validation ladder.

Everything here is read-only against:

- `C:\Users\user\dev\vLLM_for_AMD\.venv-rocm10` -- the reference `transformers` 5.17.0 /
  `torch` 2.13.0+rocm10.0.0 install. **Run every script in this directory with that venv's
  `python.exe`.** Never `pip`/`uv install` into it -- these scripts only import from it.
- `C:\AI\models\Qwen3.8-27B` -- the Qwen3.8-27B checkpoint. Shards may still be downloading; every
  tensor read from it is allowed to fail and falls back to deterministic seeded random init
  instead (see "Weight fallback" below). As of this writing all 18 shards are fully downloaded and
  every script below loads real weights.

Nothing in this directory writes outside the `--out-dir`/`--out` path you pass it (default:
a `golden_out*`/`kv_calibrate_out` subdirectory of `tools/reference/` itself, gitignored -- see
`tools/reference/.gitignore`).

## GPU rule

Per this repo's rule (only HIP device 1, the headless R9700, may be used), every script's
`--device cuda` path refuses to run unless `$env:HIP_VISIBLE_DEVICES` is exactly `'1'`:

```powershell
$env:HIP_VISIBLE_DEVICES = '1'
```

Set that once per shell before any `--device cuda` invocation. `--device cpu` always works (no
GPU rule to enforce, no device to serialize against) and is what `tests/reference/test_manifest.py`
uses.

If you run both scripts in the same shell, run them one after another, not in parallel -- this
repo's GPU tests are serialized by convention (one process on device 1 at a time).

## layer_golden.py

```powershell
$env:HIP_VISIBLE_DEVICES = '1'
C:\Users\user\dev\vLLM_for_AMD\.venv-rocm10\Scripts\python.exe tools\reference\layer_golden.py `
    --device cuda --out-dir tools\reference\golden_out
```

Builds, one at a time, straight from `transformers`' `modeling_qwen3_5.py`:

- **`layer_000_gdn`**: layer 0, a GDN (linear-attention) layer, real weights.
- **`layer_003_full_attention`**: layer 3, a full-attention layer, real weights.
- **`final_norm_lm_head`**: `text.final_norm` + a `--tiny-vocab` (default 256) row-slice of
  `lm_head.weight` (never materializes the full `[248320, 5120]` matrix -- reads only the rows it
  needs via `safetensors`' `get_slice`).
- **`mtp`**: the MTP block's inner decoder layer (`mtp.layers.0.*`, real weights -- confirmed by
  inspection to be one `full_attention`-type `Qwen3_5DecoderLayer`, no `linear_attn.*` keys), plus
  the four un-exercised fusion tensors (`mtp.fc.weight`, `mtp.norm.weight`,
  `mtp.pre_fc_norm_{embedding,hidden}.weight`) dumped raw. **transformers 5.17.0 has no MTP
  forward implementation to run** (`Qwen3_5PreTrainedModel._keys_to_ignore_on_load_unexpected =
  [r"^mtp.*"]`), so the fc/embedding-fusion math itself is not exercised here -- see the `mtp`
  component's `architecture_assumption` field in the manifest and docs/validation.md.

Each component runs a fixed-seed random `[T=64, hidden=5120]` hidden-state input through a
**prefill** call (no cache) and a **decode** call (`T=4` new tokens, continuing the prefill's
`Cache`), and dumps every intermediate the task cares about. Output:
`<out-dir>/<component>.safetensors` (one file per component) + `<out-dir>/manifest.json` (shapes,
dtypes, weight provenance, per-component status, and the tolerance table `src/model`'s tests should
use).

For a **GDN (linear-attention) layer**, captured via monkeypatching the module-level free functions
`Qwen3_5GatedDeltaNet.forward` calls into (see `GdnCapture` in the script) plus forward hooks:

- `gdn_q` / `gdn_k` / `gdn_v` / `gdn_g` / `gdn_beta`: the RAW tensors as passed to/from
  `torch_{chunk,recurrent}_gated_delta_rule` -- **pre-l2norm, pre-(1/sqrt(K))-scale,
  POST-repeat_interleave** (GQA-expanded from `Hg=16` up to `H=48` heads for q/k), and `gdn_g` is
  **per-token**, not chunk-summed. (An earlier version of this README and the script's own
  docstring incorrectly called these "post-l2norm/repeat-interleave" -- l2norm and scaling happen
  *inside* the callee, after the point these are captured.)
- `gdn_q_l2` / `gdn_k_l2` / `gdn_g_chunk_cumsum` / `gdn_core_attn_out`: the **r4d-shaped** goldens --
  l2-normed (but NOT scaled by `1/sqrt(K)`, which `r4d_gdn_chunk_scan_k128_v128_c64_bf16` takes as
  its own `scale` argument) q/k de-interleaved back to `[T, Hg=16, K=128]`, g cumulatively summed
  within each 64-token chunk `[T, H=48]`, and the chunked-scan/recurrent-rule's raw `[T, H=48,
  V=128]` output before the gated RMSNorm -- exactly the layouts `r4d_gdn_conv_prep_w4_h128_bf16`
  emits and `r4d_gdn_kkt_solve_k128_c64_bf16` / `r4d_gdn_chunk_scan_k128_v128_c64_bf16` consume/
  produce (see `r4d.h`).
- `gdn_conv_out_fn_full` / `gdn_conv_out_fn_trimmed`: the causal-conv1d output. `_full` is the raw,
  still-cache-prefixed return value of `causal_conv1d_fn` (which does NOT trim to just the new
  tokens); `_trimmed` replicates the extra slice `Qwen3_5GatedDeltaNet.forward` itself applies
  afterward (`mixed_qkv[:, :, -seq_len:]`) and has exactly `prefill_len`/`decode_len` columns for
  its stage -- **only `_trimmed` is the correct golden shape** for
  `r4d_gdn_conv_update_w4_h128_bf16`'s output. (A prior version of this tool recorded only the
  untrimmed tensor, under a stage tag that didn't even track which conv function ran -- fixed.)
- `gdn_z` / `gdn_a_raw` / `gdn_b_raw`: raw `in_proj_z`/`in_proj_a`/`in_proj_b` outputs (the gate and
  the pre-sigmoid/pre-softplus `a`/`b` that feed `r4d_gdn_recurrent_update_k128_v128_bf16_fp32state`
  alongside `A_log`/`dt_bias`).
- `gdn_gated_norm_out` / `gdn_out_proj_out`: the gated-RMSNorm output (golden for
  `r4d_gdn_gated_rmsnorm_h128_bf16`) and the post-`out_proj` tensor.

For a **full-attention layer** (also used by `mtp`'s inner decoder layer, which is
`full_attention`-typed), captured via `nn.Module` forward hooks plus a monkeypatch of
`apply_rotary_pos_emb` (same technique as `kv_calibrate.py`):

- `attn_qg_raw` / `attn_k_raw` / `attn_v_raw`: raw `q_proj`/`k_proj`/`v_proj` outputs.
- `attn_q_normed` / `attn_k_normed`: **pre-rope** (q_norm/k_norm are applied before rope in
  `Qwen3_5Attention.forward`) -- despite the name, these are NOT post-rope.
- `attn_q_post_rope` / `attn_k_post_rope`: the actual **post-rope** q/k (shape `[heads, T,
  head_dim]`, post-transpose) -- what the fp8 paged KV cache stores for K. (A prior version of this
  tool didn't capture this at all, despite the README claiming it did.)
- `attn_gate_sigmoid`: `sigmoid(gate)` from the fused `q_proj` output's second half.
- `attn_post_gate`: the tensor right before `o_proj` (i.e. `attn_output * sigmoid(gate)`), to
  isolate the `attn_output_gate` path.
- `attention_output`: the layer's `self_attn` module output (post-`o_proj`).

**Weight fallback**: for any component, if the checkpoint doesn't have the tensors this script
needs (shard not downloaded, or -- for `mtp` -- genuinely absent), the module is left at its
PyTorch-default seeded random init (`torch.manual_seed(seed)` right before construction) instead
of crashing. The manifest's `weights_source` field says which happened
(`"safetensors"` or `"random_init(seed=...)"`) -- check it before trusting a component's numbers
as a real-weights golden.

**CPU / smoke mode**: `--device cpu --force-random-init` skips the checkpoint entirely (uses
`transformers`' own default `Qwen3_5TextConfig()`, smaller dims) -- this is what
`tests/reference/test_manifest.py` drives, no checkpoint or GPU needed. Whenever a component falls
back to random init, every `Qwen3_5RMSNorm`/`Qwen3_5RMSNormGated` weight under it is perturbed with
small seeded noise (`manifest.json`'s `norm_weights_perturbed` / `final_norm_weights_perturbed`
fields say so) -- `transformers`' own default init for these weights is exactly zero, which makes
the norm's `output * (1.0 + weight)` scale identically `1.0` and can't catch a missing/incorrect
`(1 + w)` term, a classic port bug for this norm convention. Without the perturbation, the
random-init smoke path would silently never exercise that scale.

Runtime: ~20-30s on HIP device 1 (real weights, default `--prefill-len 64 --decode-len 4
--tiny-vocab 256`) for all four components combined; well under a second in `--device cpu
--force-random-init` smoke mode.

Key options (see `--help` for the rest): `--gdn-layer` / `--attn-layer` (default 0 / 3),
`--prefill-len` / `--decode-len` (default 64 / 4), `--tiny-vocab` (default 256), `--seed` (default
1234), `--skip-mtp`.

## vision_golden.py

```powershell
$env:HIP_VISIBLE_DEVICES = '1'
C:\Users\user\dev\vLLM_for_AMD\.venv-rocm10\Scripts\python.exe tools\reference\vision_golden.py `
    --device cuda --out-dir tools\reference\golden_out
```

The vision-tower analogue of `layer_golden.py` -- see `docs/vision.md` for the full architecture
spec this script validates against. Builds `Qwen3_5VisionModel` directly from `modeling_qwen3_5.py`,
loads real `model.visual.*` weights (333 tensors, all present in the real checkpoint), preprocesses
a real image with the checkpoint's own configured `transformers.AutoImageProcessor`
(`Qwen2VLImageProcessor`), and dumps `pixel_values`, every encoder block's output (27), block 0's
full internal chain (norm1/qkv-raw/post-rope-q-k/proj-out/norm2/mlp-fc1/mlp-fc2), and the merger
output. `--image <path>` uses a real photo; without it, a deterministic synthetic 448x448 test image
is generated and saved to `<out-dir>/vision_test_image.png` for reproducibility. Output:
`<out-dir>/vision_tower.safetensors` + `<out-dir>/vision_manifest.json` (same tolerance-table
convention as `layer_golden.py`'s `manifest.json`, kept in a separate file since this rung has no
text-layer components to share a manifest with).

**Found and fixed while building this script**: `common.py`'s `ShardIndex.get_tensor`/
`get_row_slice` had a real dangling-mmap bug (see `common.py`'s inline comment and `docs/vision.md`)
that a tight loop over ~300+ tensors reproducibly turns into a Python interpreter crash (access
violation) -- `layer_golden.py`'s/`kv_calibrate.py`'s much smaller per-component tensor counts never
triggered it. Fixed with `.clone()`; every script in this directory benefits, no other script's own
code changed.

Runtime: ~15-25s on HIP device 1 (real weights, one 448x448 test image, 27 blocks).

## kv_calibrate.py

```powershell
$env:HIP_VISIBLE_DEVICES = '1'
C:\Users\user\dev\vLLM_for_AMD\.venv-rocm10\Scripts\python.exe tools\reference\kv_calibrate.py `
    --device cuda --layer 3 --out tools\reference\kv_calibrate_out\kv_descale.json
```

Prototype of the static fp8 KV descale calibration from `docs/container-format.md` ("KV descale
tables"). Tokenizes a calibration text (real tokenizer, via `AutoTokenizer` -- this only *uses*
the tokenizer to build input ids, it doesn't touch anything under `src/tokenizer` or
`tools/reference/tok_golden.py`, which belong to the tokenizer agent), gathers the calibration
tokens' `text.embed_tokens` rows (real weights, row-gathered so the full `[248320, 5120]` table is
never materialized), runs them through `--layer`'s `input_layernorm` + `self_attn` (real weights),
and records the per-kv-head `amax` of K (**post-rope** -- captured by monkeypatching
`apply_rotary_pos_emb`, matching exactly what the paged fp8 cache would store) and V (no rope
applied to V; captured via a forward hook on `v_proj`). `--layer` must be a `full_attention` layer
(0-indexed `3, 7, 11, ...`; the script asserts this and explains why if you pick a GDN layer --
r4d's paged fp8 KV cache is attention-only).

**Calibration corpus**: `--calib-text-file <path>` (default: `tools/reference/calib.txt` next to
this script if present, else the old built-in `DEFAULT_CALIBRATION_TEXT` paragraph). `calib.txt` is
a deliberately mixed English-prose / source-code / Thai-prose corpus (a few hundred tokens after
repetition to `--num-tokens`) -- a single-register corpus (e.g. English news prose alone) tends to
under-estimate the true per-head amax against the token distributions (code, Thai, mixed-script)
this engine will actually see, which would make the calibrated descale too small and clip outlier
activations at inference time. The output JSON records `calib_text_source` (which file was used)
and `calib_text_sha256` for provenance.

Output JSON: `{"<layer_idx>": {"k_amax": [4 floats], "v_amax": [4 floats], ...}}` plus provenance
fields (`weights_source`, `embed_source`, `config_sha256`, `torch_dtype`, `device`,
`calib_text_source`, `calib_text_sha256`, etc. -- `torch_dtype`/`device` matter because `amax` is
computed in bf16 on `--device cuda` and fp32 on `--device cpu`, materially different numbers) and
two prose fields baked into every run's output: `"caveat"` (this is a **prototype** -- it feeds the
calibration tokens' raw embeddings straight into one layer, skipping every preceding layer's
transform, so the hidden-state distribution isn't what that layer truly sees mid-stack) and
`"converter_consumption"` (exactly how `src/convert` turns `k_amax`/`v_amax` into
`text.layers.{i}.attn.k_descale`/`.v_descale`).

**Descale convention (exact)**: `descale = amax / 448.0`, where `448.0` is the OCP e4m3fn finite
max magnitude and `amax` is the per-kv-head absolute-max over the calibration set (K: post-rope;
V: raw `v_proj` output, no rope). This matches `src/kernels/src/r4dx_kernels.hip`'s
`r4dx_kv_write_paged_fp8_hnd`, which writes `stored_fp8 = fp8e4m3(real_bf16_value / descale[head])`
-- i.e. it *divides* by descale before the fp8 cast -- so the inverse, `dequant = fp8_value *
descale[head]`, is exactly what the attention kernel must (and does) use to reconstruct the real
value for QK^T / PV. `r4dx-convert --kv-calib <json>` (`src/convert/main.cpp`) applies this same
`amax / 448.0` formula per kv head when it fills `text.layers.{i}.attn.k_descale` / `.v_descale`,
replacing `docs/container-format.md`'s placeholder `1.0`; a layer missing from the calibration JSON
(or with a `k_amax`/`v_amax` of the wrong length) falls back to `1.0` with a `WARNING` printed to
stderr, not a hard error. The resulting per-layer `fp32[kv_heads]` vector must still be broadcast to
`r4d.h`'s runtime `(num_seqs, kv_heads)` shape, one identical row per sequence, by whatever in
`src/model` builds `R4DArgs.k_descale`/`.v_descale` -- calibration is not per-sequence.

Running the script again with the same `--out` path **merges** the new layer's result into the
existing file instead of overwriting it, so calibrating all 16 full-attention layers is a matter of
running `--layer 3`, `--layer 7`, ... `--layer 63` in sequence against the same `--out`. A
`--calib-text-file` argument (default `tools/reference/calib.txt`) picks the calibration corpus; see
above.

Runtime: ~10s on HIP device 1 (default `--num-tokens 256`); a few seconds on CPU.

## tok_golden.py

Not this component's -- owned by the tokenizer agent. Don't add it here.

## Tests

`tests/reference/test_manifest.py` is CPU-only, needs no checkpoint, and has **no pytest
dependency** (a plain script with bare asserts and a `__main__` entry, since the reference venv --
the only python on this machine with `torch`+`transformers` importable together -- doesn't have
`pytest` installed and is read-only). Run it directly:

```powershell
C:\Users\user\dev\vLLM_for_AMD\.venv-rocm10\Scripts\python.exe tests\reference\test_manifest.py
```

It's also registered as ctest test `reference_manifest` in `tests/CMakeLists.txt`, so
`.\tests\run_tests.ps1`'s `ctest --preset win-hip` picks it up automatically alongside the C++
tests once the rest of the project configures cleanly. It checks every component ran `status=ok`
with honest `random_init` provenance, that each `.safetensors` file was written and non-empty, the
full expected tensor-name set per component (including the r4d-shaped GDN tensors and post-rope
attention tensors above -- a regression that silently drops one now fails this test instead of only
being noticed by whoever diffs against the missing tensor), that the decode GDN conv output's
trimmed length matches `--decode-len`, and that the manifest's tolerance table is well-formed.
