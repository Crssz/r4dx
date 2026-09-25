# Quantization v2 -- error-feedback rounding, residual rotation, per-tensor groups

Branch `quant2` (worktree `%USERPROFILE%\dev\r4dx-quant2`, from `main` at `e18a3a9`). Written
2026-09-25. Three ideas borrowed from exllamav3's EXL3 pipeline, adapted to r4dx's existing 4-bit
layouts and kernels. Where this says MUST, do exactly that; anything not covered is a question for
the user, not a judgment call.

Why now: `docs/validation.md` "Milestone 11 / recipe" ends with v6 at mean KL **0.03851** / top-1
**90.93%**, an honest ceiling of ~0.0320 for per-class promotion, and llama.cpp `Q4_K_M` at
**0.011-0.014** on the same checkpoint. Its conclusion was that closing the gap needs "a better 4-bit
*scheme*". exllamav3 gets most of its quality from *how values are chosen* (Hessian-aware rounding
with error feedback, on incoherence-processed weights) rather than from its storage format -- and
r4dx's `--quant` slot already changes values without touching bytes. Q1 below tests that cheaply
before any kernel work.

---

## 0. Decisions at a glance

| Question | Decision | Why (short) |
|---|---|---|
| Phase order | **Q1** LDLQ (converter only) -> **Q2a** residual rotation (converter + 2 online ops) -> **Q2b** online Hadamard on down/o/out_proj inputs (3 kernels) -> **Q3** per-tensor w4a16 group | cheapest first; each later phase is measured on top of the previous one, so each gets its own KL number |
| Q1 algorithm | GPTQ/LDLQ: `U = chol(H^-1)` upper, column-by-column rounding with error feedback onto later columns, lazy 128-column block updates | the byte layout is unchanged (`q`/`scale`/`zero` per `(row, group)`), so kernels, loader, container, TP and speed are untouched |
| Q1 per-group scale | chosen at the group's first column from the *current* (error-updated) weights, by the existing `--quant search` grid weighted by `diag(H)`, **without** its final refit | reuses the measured Milestone 10 search; the refit assumes fixed codes, which LDLQ does not have |
| Q1 Hessian | `H = mean over tokens of x x^T`, bf16 checkpoint activations (not sequential/quantized), fp64 accumulate, fp32 upper triangle on disk, one file per distinct linear input | the reference forward already exists; sequential GPTQ is a later refinement, not v1 |
| Q1 corpus | existing calib (8,316 tok) + 64 x 2048 WikiText-2 train + 16 x 2048 repo source, disjoint from `kl_corpus/` | `mlp.down` has K = 17,408; 8,316 rows cannot make its Hessian full-rank |
| Q1 math location | C++ in `r4dx-convert` (CPU, AVX-512 microkernel, deterministic) | converter stays the single, reproducible, GPU-free producer of containers |
| Q2a rotation | one orthogonal `Q` on the 5120-wide residual stream, folded into every in-projection (K side) and out-projection (N side); norm weights folded into the next linear; **online `Q` at stack entry and `Q^T` at stack exit** | embedding table, vision rows, lm_head, MTP and the DFlash2 drafter stay byte-identical and un-rotated |
| Q2a `Q` | `Q = D . (I_5 (x) H_1024/32) . (R_5 (x) I_1024)`: random signs, 5 blockwise normalized Hadamards, a random 5x5 orthogonal mix | 5120 = 5 x 1024 has no Hadamard; this is exact, orthogonal and O(n log n) |
| Q2b online Hadamards | `mlp.down` input: block 512, fused into `silu_mul`; `attn.o` input: block 256 per head, fused into the output-gate multiply; `gdn.out_proj` input: block 128 per head, one extra kernel | block sizes divide both the full and the TP=2 per-rank K, so no block straddles a rank |
| Q3 | libr4d `r4d_gemm_w4a16_nt_m64` group becomes a template parameter (32/64/128 instantiated); group recorded per tensor; allocation by the Milestone 11 nats-per-GiB rule, with bf16 keeps (and un-keeps of the recipe's attn.k/v) priced in the same currency (5.3) | makes depth/class-aware precision possible without a second format |
| GPU work | every GPU run is handed to the user as an exact command (repo rule); CPU/convert work proceeds without asking | `gpu-runs-need-approval` |

---

## 1. Goals, non-goals, gates

### 1.1 Goals

1. Lower mean KL vs the bf16 reference at **equal or lower weight bytes** than v6
   (`weights=16.4065 GiB`), measured by rung 4 (`tool_teacher_forced_logprobs` + `kl_report.py` on
   `tools/reference/kl_corpus/`), `--layout w4a16`.
2. Decode throughput within 1.5% of v6 for Q1 and Q2a (both are byte-layout-neutral and add at most
   two tiny kernels per token), within 3% for Q2b.
3. Every existing feature keeps working on a rotated container: plain/sampled decode, chunked
   prefill, MTP, DFlash2, vision, TP=2, the server.

### 1.2 Non-goals (v1)

- A new storage format (trellis/QTIP) -- see the comparison notes this design came from.
- Sequential GPTQ (Hessians from already-quantized preceding layers).
- Act-order (column permutation by `diag(H)`): it breaks contiguous K groups.
- Rotating the MTP head, the DFlash2 drafter, lm_head's input, or the vision tower.
- LDLQ for the DFlash2 drafter (no Hessians are captured for it).

### 1.3 Gates

| Gate | Phase | Pass condition |
|---|---|---|
| G1 | Q1 | `convert_quant_ldlq` ctest: (a) `H = I`, damp 0 reproduces the no-refit group search byte for byte; (b) on correlated random inputs the proxy loss `tr((W-Wq) H (W-Wq)^T)` is <= 0.8x the `search`+`diag(H)` quantizer's, for all three layouts at both w4a16 groups (the test gates its own fixture at 0.2x, and also matches every layout against an unblocked per-row GPTQ reference: <= 0.1% of codes differ, proxy within 0.5%); (c) output bytes identical at 1, 3 and 16 threads; (d) `SubMatMul`/Cholesky/inverse match scalar references to 1e-4 relative |
| G2 | Q1 | Hessian capture: every quantized linear covered, every H symmetric/finite/PSD-ish (min diag > 0), rows >= 4 x K for every file, shared-input check passes |
| G3 | Q1 | **pilot**: v6 recipe + `--ldlq "mlp\."` -- KL must fall by >= 10% vs v6 (0.03851 -> <= 0.0347). **Stop rule: if it does not, stop Q1 and report before Q2.** |
| G4 | Q1 | full LDLQ container: KL, top-1, decode tok/s (plain and `--dflash k=7`) recorded; tok/s within 1.5% of v6 |
| G5 | Q2a | bf16 rotated container vs bf16 unrotated: rung-4 KL <= 2x the reference's own self-noise (3.9e-4 nats); then quantized+LDLQ: KL lower than G4's |
| G6 | Q2a | `ctest` green; `tools/validate_dflash.ps1`, `tools/server/smoke.ps1 -Vision -Dflash`, TP=2 `tp1_identity`-style check on the rotated container |
| G7 | Q2b | KL lower than G5's quantized number AND decode cost <= 3% vs v6 |
| G8 | Q3 | at weight bytes <= v6, KL lower than the best of G4/G5/G7 |

---

## 2. Q1 -- LDLQ error-feedback rounding

### 2.1 Hessian capture: `tools/reference/hessian_capture.py`

Layer-major, unlike `imatrix_capture.py`: all calibration sequences are embedded once, then for
each layer `i` the layer is materialized once and every sequence is run through it
(`hidden_states=h, position_embeddings=(cos, sin), attention_mask=mask or None` exactly as
`StreamingReference._forward_manual`). This holds one layer's accumulators at a time
(<= 3.2 GiB fp64) instead of all 64 layers'.

Distinct inputs ("taps") per layer, each hooked on ONE representative module:

| layer type | tap | representative module | container keys served |
|---|---|---|---|
| GDN | `in` | `linear_attn.in_proj_qkv` | `gdn.in_proj_qkv`, `gdn.in_proj_z` |
| GDN | `out` | `linear_attn.out_proj` | `gdn.out_proj` |
| attention | `in` | `self_attn.q_proj` | `attn.qg`, `attn.k`, `attn.v` |
| attention | `out` | `self_attn.o_proj` | `attn.o` |
| both | `mlp_in` | `mlp.gate_proj` | `mlp.gate_up` |
| both | `mlp_mid` | `mlp.down_proj` | `mlp.down` |

plus `lm_head` (post-final-norm hidden) and, unless `--no-mtp`, the MTP head's four taps via
`imatrix_capture.MtpTaps` (duck-typed accumulators). A shared-input gate hooks `in_proj_z`,
`k_proj`, `v_proj` and `up_proj` on the first layer of each type and MUST see the same storage as
the representative.

File format, one per tap, `<out-dir>/<tap-id>.hess`:

```
[8]  magic "R4DXHES1"
[4]  u32 K
[4]  u32 flags            (bit 0 = packed upper triangle; always set in v1)
[8]  u64 rows             (tokens accumulated)
[8]  f64 trace            (sum of diag, for a cheap integrity check)
[32] reserved (zero)
[..] f32 H[i][j] for i in 0..K-1, j in i..K-1   (row-major packed upper triangle, H = X^T X / rows)
```

`<out-dir>/hessian.json` maps every container base name to its tap file, records K and rows per
file, the corpus (names, sha256, token counts), the checkpoint's `config.json` sha256, the
converter audit (reused from `imatrix_capture.audit_converter_source`) and the gate results.

Corpus (defaults): `calib.txt` + `kv_calib_corpus/` (as today) + the first 64 non-overlapping
2048-token windows of `D:/models/wikitext-2-raw/wiki.train.raw` + 16 x 2048 tokens of this repo's
C++/HIP/Python sources, **excluding** `src/model/model.cpp` and
`tools/reference/layer_golden.py` (the `kl_corpus/` excerpts' sources). ~172k tokens. Disk:
~51 GiB for the 64 layers (the down tap is 578 MiB packed), + lm_head + MTP. The script prints the
estimate and refuses to start if the target drive lacks 1.1x of it.

### 2.2 Converter: `quant_ldlq.hpp`, `dense_linalg.hpp`, `hessian_store.hpp`

`dense_linalg.hpp` (row-major float32, deterministic: parallel only over output rows, fixed
reduction order, FMA in both the AVX-512 and the scalar path):

- `SubMatMul(M, N, K, A, lda, B, ldb, C, ldc, threads)`: `C -= A.B`.
- `CholeskyLower(A, n, threads)` (blocked, 128) -> `bool` (false on a non-positive pivot).
- `InvertLower(L, n, threads)` (blocked) in place.

`U = chol(H^-1)` upper without forming `H^-1`: with `J` the reversal permutation, factor
`J H J = L L^T`; then `U = J L^-1 J`. (Proof: `H = R R^T` with `R = J L J` upper, so
`H^-1 = R^-T R^-1`, and `R^-1` is upper with a positive diagonal, hence the unique upper Cholesky
factor of `H^-1`.)

Damping: `H += damp . mean(diag(H)) . I`, `damp` from `--ldlq-damp` (default 0.01). A failed
Cholesky retries at 10x damp, twice, logging each retry; a third failure is a hard error.

LDLQ loop, per tile of `R = 128` rows (tiles are independent, one per thread), blocks of `B = 128`
columns (`B % group == 0` for every layout's group, so a group never straddles a block):

```
for block [b0, b1):
  copy W_tile[:, b0:b1] -> Wb   (R x 128)
  for i in b0..b1-1:
    if (i - b0) % group == 0: choose per-row (scale, zero) of group [i, i+group) from Wb's current
                              values (search grid, weights diag(H)[i..], NO refit)
    q[:, i] = round per layout;  e = (Wb[:, i] - dequant(q[:, i])) / U[i][i]
    Wb[:, i+1:b1] -= e (x) U[i, i+1:b1];  E[:, i-b0] = e
  W_tile[:, b1:] -= E . U[b0:b1, b1:]       (SubMatMul)
```

Per layout: w4a16 asymmetric (free zero), w4a8 zero pinned at 8, mxfp4 per-32 E8M0 exponent
(round-up vs round-up-minus-one, as `QuantizeMxfp4Search`) then `EncodeE2M1`. Outputs are exactly
the vectors `QuantizeInt4Asymmetric` / `QuantizeInt4SymmetricPinned8` / `QuantizeMxfp4` produce, fed
to the unchanged packers.

`hessian_store.hpp`: reads `hessian.json` + `.hess`, validates magic/K/trace, expands to a full
symmetric `K x K`, and caches the last factorization (keyed by file and damp) so the linears that
share a tap (`in_proj_qkv`/`in_proj_z`, `qg`/`k`/`v`) factor once.

### 2.3 CLI

`r4dx-convert ... --hessian-dir <dir> --ldlq <regex> [--ldlq-damp 0.01]`

- `--ldlq <regex>` selects the container base names quantized by LDLQ (`regex_search`, like
  `--keep-bf16`); every other linear follows `--quant`/`--imatrix` as today. `".*"` = everything.
- `--ldlq` requires `--hessian-dir`; a matching linear with no Hessian is a hard error before the
  first shard is read (the manifest is checked during planning).
- `--keep-bf16` wins over `--ldlq` for a linear both match.
- `__metadata__.r4dx_convert_run` records `ldlq` (pattern), `ldlq_damp`, `hessian_dir`, the
  manifest sha256 and the list of LDLQ'd bases. A rotated container also records
  `ldlq_rms_linears`, the in-projections rounded against the weightless rms Hessian (section 3.2).

Expected cost: ~2e14 multiply-adds for the whole model per layout (lm_head alone 6.5e12). At the
measured microkernel rate (reported by the ctest) this should be a few minutes per layout on the
9950X; the converter logs per-linear time.

---

## 3. Q2a -- residual-stream rotation

Math (row-vector convention, `x` a residual row): pick orthogonal `Q` (5120 x 5120). Inside the
64-layer stack the residual is `x' = x Q`. RMSNorm without a weight commutes with `Q`; a zero-centred
norm `rms(x) (1 + w)` does not, so its `(1 + w)` is folded into the next linear and the stored norm
weight becomes 0. For a linear `y = x_n W^T`:

| tensor | fold |
|---|---|
| `input_layernorm`, `post_attention_layernorm` | stored as 0 (kernel computes `rms(x) . 1`) |
| every in-projection (`gdn.in_proj_qkv/z/a/b`, `attn.qg/k/v`, `mlp.gate_up`) | `W' = W diag(1 + w_norm) Q` (K side) |
| every out-projection (`gdn.out_proj`, `attn.o`, `mlp.down`) | `W' = Q^T W` (N side) |
| everything else (conv1d, A_log, dt_bias, q/k norms, descales, gdn norm) | unchanged: they act after a projection, inside heads |
| `text.embed_tokens`, `text.final_norm`, `lm_head`, `mtp.*`, `vision.*`, the DFlash2 drafter | **unchanged** |

Online ops (`src/kernels/rotate_residual.hip`, one workgroup per row, 5120 floats in LDS): `Q` on
the stack's input rows after embedding/vision splicing, `Q^T` on the stack's output rows before
`final_norm`/MTP priming, and `Q^T` on DFlash2 feature captures (the drafter consumes un-rotated
features). Every stack entry and exit (plain, chunked prefill, verify window, MTP priming, DFlash
feature capture, TP ranks) MUST be enumerated in the Q2a implementation note and covered by the
bf16-rotated-vs-unrotated gate G5.

`Q` is generated from a seed recorded in `__metadata__.rotation` (`{"kind": "hadamard1024x5",
"seed": ...}`); the loader refuses a rotated container on a binary without the online op, and a
binary never applies the op to an unrotated container. The fold happens in fp32 before
quantization; bf16 linears (`gdn.in_proj_a/b`, `--keep-bf16` ones) are rounded once from the
folded fp32. The Hessian capture for a rotated container MUST record activations in the rotated
basis (`hessian_capture.py --rotation-seed`): `H' = Q^T H Q` for residual-side taps, computed from
`H` exactly rather than re-captured. (As built, the converter does this change of basis itself,
from an unrotated capture; for the norm-fed in-projections it needs the weightless rms taps of
section 3.2.)

### 3.1 Q2a implementation note (runtime)

The metadata kinds the converter writes are `q2a` and `q2ab` (docs/container-format.md "Residual
rotation"); `hadamard1024x5` above was a draft name, and the loader refuses it like any other
unknown kind. This note covers the runtime half. Q2b's three input Hadamards are listed here too,
because they share the loader and the call sites.

**Off by default.** Everything hangs on `Container::HasRotation()`, which is true only when
`__metadata__.rotation` is present. Without the key, `Model::RotateResidual` returns before launching
anything and `Model::HadSigns()` returns three null pointers. Every layer call then takes its
pre-quant2 branch: the same kernels in the same order, the same r4dx launch count, and the same
profile span names. The runtime has no flag for this: the container decides.

**Loader** (`src/model/rotation_meta.h`, `src/model/container.{h,cpp}`, `src/model/tp/tp_shard.cpp`):

- `rotation_meta.h: ParseRotationMetadata` is HIP-free and header-only. Both `Container::Load`
  (TP=1) and `Container::LoadShard` (TP) call it right after the config is parsed, before any upload.
  It returns nullopt when the key is absent. It throws, naming the path, in these cases:
  - an unknown `kind`;
  - a missing, mistyped or negative `seed`;
  - `hidden`/`block` other than 5120/1024, or a model `hidden_size` other than 5120;
  - a q2a block that carries `had`;
  - q2ab `had` blocks other than 512/256/128;
  - a q2ab model with `head_dim != 256`, `linear_value_head_dim != 128`, or an
    `intermediate_size` that is not a multiple of 512. The o and gdn_out Hadamards run one head per
    block.

  `RotationTensors(spec, cfg)` lists the tensors and fp32 lengths a config needs.
- `container.cpp: LoadRotationWeights` runs after `lm_head` on both paths. For each tensor it:
  - checks presence and the global length on disk;
  - checks on the host that every sign is exactly ±1 and that `mix5` is orthogonal to 1e-5;
  - uploads it (`UploadRawF32` at TP=1, `ShardLoader::Raw<float>` under TP);
  - checks the uploaded length against the rank config.

  The result is `RotationWeights` (`container.h`), exposed as `Container::Rotation()`. One stderr
  line announces a rotated container.
- `tp_shard.cpp: RuleFor`:
  - `rotation.signs` and `rotation.mix5` replicate.
  - `rotation.had_down_signs` is `Rows({intermediate_size})`, `rotation.had_o_signs` is
    `Rows({heads*head_dim})`, and `rotation.had_gdn_out_signs` is `Rows({ValueDim()})`. Each is one
    segment, so rank r gets `[r*K/2, K/2)`: exactly its linear's `RankCols`, a whole number of
    blocks (mlp.down 17 x 512, attn.o 12 heads, gdn.out_proj 24 heads).

  Any other `rotation.*` name still throws.
  `tests/model/test_rotation_meta.cpp` (CPU) checks that equality for both ranks.

**The online ops.** All of them are enqueued on `stream_` and none is a new kind of barrier. Each
site is listed as file: function.

| # | site | op | rows | why here |
|---|---|---|---|---|
| E1 | `model.cpp: Model::RunChunk`, after the embed gather (device or host) and `SpliceImageEmbeddings`, before the positions upload and layer 0 | `x Q` | `buf_a_[0, T)` | the one entry for Prefill, PrefillMultimodal (and its text-only fallback), DecodeStep, DecodeStepGreedy, DecodeStepSampled (and its fallbacks), TpWarmup and every TP rank. Image rows are spliced first, so they are rotated along with the text rows |
| E2 | `model.cpp: Model::VerifyWindow`, after the embed gather | `x Q` | `buf_a_[0, T)` | every speculative verify: VerifyAndResolveRound from the MTP and DFlash2 decode paths, the TpWarmup speculative round, and direct callers. It has no splice, because candidates always follow the prompt |
| E3 | `model.cpp: Model::DecodeStepProfiled`, after the `embed` span | `x Q` | row 0 | span `rotate.entry` |
| E4 | `model.cpp: Model::PrefillProfiled`, per chunk, after the `embed` span | `x Q` | `[0, T)` | span `rotate.entry`. Needed for correctness: the KV/GDN state written here is read by later decode steps |
| X1 | `model.cpp: Model::RunChunk`, right after the layer loop | `x Q^T` | `cur[0, T)` | before every reader of the pre-final-norm residual: MTP within-chunk `PrimeKv` (rows 0..T-2), the `mtp_seed_hidden_` copy (row T-1, read later by the boundary `PrimeKv`, `MtpHead::Draft` and `DebugSeedHiddenBf16`), and `FinalLmHead` |
| X2 | `model.cpp: Model::VerifyWindow`, right after the layer loop | `x Q^T` | `cur[0, T)` | before `FinalLmHead` (all T rows) and `mtp_last_hidden_ = cur`, which DecodeStepMtpImpl copies into `mtp_seed_hidden_` |
| X3 | `model.cpp: Model::DecodeStepProfiled`, after the loop | `x Q^T` | row 0 | span `rotate.exit`, before the `final_norm+lm_head` span |
| X4 | `model.cpp: Model::PrefillProfiled`, per chunk, after the loop | `x Q^T` | `[0, T)` | span `rotate.exit`. Nothing reads the result; it keeps the profile at RunChunk's real cost |
| D1 | `model.cpp: Model::RunChunk`, after X1, gated on `dflash_capture_active` | `x Q^T` | `dflash_features_dev_[0, T*cols)` | the captures are layer INPUTS, so they are rotated. They must be derotated before the observer, `dflash_->InjectFeatures` and `DflashFeatureBuffer()` read them (all after the synchronize). The gate is the capture's own: a chunk that did not capture must not rotate an earlier chunk's rows a second time |
| D2 | `model.cpp: Model::VerifyWindow`, after X2, gated on `!dflash_target_layers_.empty()` | `x Q^T` | `dflash_features_dev_[0, T*cols)` | the same, with that capture's own gate, which is not `dflash_injection_enabled_`. Read by DecodeStepDflashImpl's `InjectFeatures` and by tests |
| H1 | `mlp.cpp: Mlp::Forward` (`down_had_signs`, a ctor argument) | `(silu*up) Hb`, B 512 | `[T, intermediate]` | the backbone's only `silu_mul`, fused as `r4dx_silu_mul_hadamard_bf16`, with the w4a8/mxfp4 epilogue applied to the rotated row |
| H2 | `attention_layer.hpp: AttentionLayer::Forward` (`AttnWeights::o_had_signs`) | `(o*sigmoid(g)) Hb`, B = head_dim | `[T, H*D]` | fused as `r4dx_model_attn_gate_mul_hadamard_bf16`, on the prefill and decode/verify paths alike |
| H3 | `gdn_layer.cpp: GdnLayer::Forward` (`GdnLayerParams::out_had_signs`), after the prefill/decode branches rejoin at `out_core` | `h Hb` in place, B = V | `[T, H*V]` | `r4dx_hadamard_inplace_bf16`. Decode's gated norm lives inside libr4d's recurrent kernel, so it cannot be fused |

The Hb pointers come from `Model::HadSigns()` and are set in exactly the four backbone loops (RunChunk,
VerifyWindow, DecodeStepProfiled, PrefillProfiled). All of them are per-call data with a null
default, never a Container-wide flag. `MtpHead` reuses `Mlp` and `AttentionLayer` on never-folded
weights, and it never receives the pointers.

**Why the list is complete.**

1. *Every pass through the folded weights is one of four loops.* The only constructions of
   `GdnLayer`, `AttentionLayer` and `Mlp` over `container_.Layer(i)` are in RunChunk, VerifyWindow,
   DecodeStepProfiled and PrefillProfiled. The only other constructions are in `mtp_head.cpp`
   (Draft, PrimeKv), and they use the `mtp.*` weights, which are never folded. Every public Model
   entry point that runs the backbone reaches one of those four loops. `LocalTextModel` and
   `TpModel` only forward to `Model`, per rank, and the residual is replicated under TP. So there is
   no TP-specific site: each rank rotates its own full rows with replicated `signs`/`mix5`, and
   applies Hb with its own sign slice.
2. *Each loop has one way in.* Each loop fills `buf_a_` from the embedding gather, and RunChunk then
   splices image rows. The entry op follows both. Layer 0 is called with `normed_in == nullptr`, so
   its own input rmsnorm runs on the rotated rows, and no fused kernel crosses the entry.
3. *Each loop has one way out.* The residual leaves only as `cur` after the loop. The last layer's
   `Mlp` does a plain `residual_add` (`next_norm_weight == nullptr`), so no fused residual+rmsnorm
   straddles the exit. Every reader of the residual is downstream of the exit in the same call, or
   in a later call: `FinalLmHead`, MTP priming, `mtp_seed_hidden_`, `mtp_last_hidden_`,
   `DebugSeedHiddenBf16`.
4. *Inside the stack, nothing reads x in the wrong basis.* The residual adds and the fused
   residual+rmsnorm epilogues are basis-agnostic, because the stored norm weight is 0: `rms(x)*1`
   commutes with Q. Every reader of the normed residual is a folded in-projection. The one other
   in-loop reader is the DFlash2 capture, which D1/D2 undo.
5. *Q2b.* Each fused op has one backbone launch site: `silu_mul` is launched only by
   `Mlp::Forward`, the gate multiply only by `AttentionLayer::Forward`, and the gated-norm output
   comes only from GdnLayer's two branches, which rejoin at one point.
6. *Untouched on purpose.* These keep their un-rotated behaviour, and nothing may pre-rotate them:
   - `text.embed_tokens`, both the host copy and the VRAM mirror, which the MTP head and the DFlash2
     drafter (`MakeTargetEmbeddingProvider`) also read;
   - the vision tower's output, which is spliced before the entry op;
   - `final_norm` and `lm_head`, which the drafter shares;
   - the MTP head;
   - the drafter's own `silu_mul` and attention.

**Costs and consequences.**

- *Launch count.* r4dx-owned launches per decode token rise by 2 for q2a (entry and exit). q2ab adds
  48 more on the 64-layer model: one `gdn.out_hadamard` per GDN layer. `silu_mul_hadamard` replaces
  `silu_mul` one for one, and the attention variant is uncounted, like `gate_mul`. So
  docs/perf.md's 259 becomes 261 for q2a and 309 for q2ab. RunChunk also launches one more when a
  DFlash2 capture is active.
- *Determinism.* Every rotate kernel is row-independent, so a row's result does not depend on T or
  on the grid. The TP=1 identity baseline and the verify-row exactness from f7d4927 hold for each
  kind of container separately. They do not hold across rotated and unrotated containers, because
  the weights differ.
- *Unrotated-container tests.* `test_dflash_feature_capture` compares capture column 0 bit for bit
  with the embedding rows and checks the launch count, and the layer-golden tests assume unrotated
  weights. They keep passing because they run on the unrotated 4-layer container. On a rotated
  container, column 0 is `bf16(bf16(x Q) Q^T)`, which is not bit-equal to `x`.
- *Binaries older than this branch* never read `__metadata__.rotation`. They load a rotated container
  without complaint and produce garbage. Only binaries from this branch refuse unknown kinds or apply
  the ops.

**Tests.** `test_rotation_meta` (CPU, always runs) covers the parse, all 21 refusal cases, the
model-shape refusals, and the TP slices. `test_tp_shard` gains the five `RuleFor` cases. The GPU
side is `tests/kernels/test_rotate_residual`, `test_attn_gate_mul_hadamard`, and gates G5/G6 on a
real rotated container. All of it is handed to the user to run.

### 3.2 Q2a: the in-projection Hessian comes from the weightless rms tap

A rotated container runs each zero-centred norm without its weight and folds `(1 + w)` into the next
in-projections. So their input is `r' = rms(x) Q`, where `rms(x) = x * rsqrt(mean(x^2) + eps)` has no
weight, and LDLQ must round against `H' = E[r'^T r'] = Q^T H_rms Q` with
`H_rms = E[rms(x)^T rms(x)]`. Section 2.1's taps record the post-norm input
`x_n = rms(x) (1 + w)`, so `H = D H_rms D` with `D = diag(1 + w)`. The first converter recovered
`H_rms` by division, as `Q^T D^-1 H D^-1 Q` (`rotation.hpp: TransformHessianQ`). That has two
problems:

- **It fails at a dead channel.** Where `(1 + w_j) == 0`, `H` has an all-zero row and column, and no
  `D^-1` exists. The real checkpoint has one: layer 7 `post_attention_layernorm[3994]`. Refusing is
  the right call there. The rotation spreads channel `j`'s residual value, which can be large,
  across every direction of its block. A Hessian that exempted the channel would give the direction
  `e_j Q` zero weight, and rounding error would leak along it with no penalty.
- **It amplifies noise where `|1 + w|` is merely small**, by `1 / (1 + w)^2`.

**The fix: capture `H_rms` directly.** `hessian_capture.py --rms-taps` (with a full capture) or
`--rms-only` (merged into an existing set) hooks the input of each text layer's `input_layernorm`
and `post_attention_layernorm` (a forward_pre_hook on the norm module, so `x` is the bf16 residual).
It accumulates `r = x.float() * rsqrt(mean(x.float()^2) + eps)` -- `Qwen3_5RMSNorm._norm(x.float())`
exactly, with the module's own `eps` (`Qwen3_5RMSNorm` stores it as `.eps`, not `.variance_epsilon`),
without the weight multiply or any bf16 cast -- as an fp32 GEMM per sequence into an fp64
accumulator, into `L{i:02d}.in.rms.hess` and `L{i:02d}.mlp_in.rms.hess` (same format as 2.1; the
name is the post-norm tap's with `.rms` before `.hess`). `hessian.json` gains `"rms_keys"`, which
maps each norm-fed in-projection to its file: `gdn.in_proj_qkv/z` and `attn.qg/k/v` to `L.in.rms`,
and `mlp.gate_up` to `L.mlp_in.rms`. `gdn.in_proj_a/b` stay bf16 and are never LDLQ'd, so they have
no entry.

- **Gates** (`evaluate_rms_gates`, recorded under `"rms_capture".gates`): finite; min diagonal > 0
  with NO dead-channel exemption (`rms(x)` has no structurally zero channel); `rows >= 4K` and equal
  to the paired post-norm file's rows; and consistency: on every channel with `(1 + w_i) != 0`,
  `diag(H_post)_i` within 2% of `(1 + w_i)^2 diag(H_rms)_i` (the worst channel is reported), and
  `diag(H_post)_i == 0` exactly where `(1 + w_i) == 0`. That proves both captures saw the same
  activations through the same norm: the only difference is the bf16 rounding of the post-norm
  input, <= ~0.4% on a mean square. If any gate fails, nothing rms goes into `hessian.json`; the
  report is written to `hessian_rms.failed.json` instead.
- **`--rms-only`** runs the full layer-major forward with only the rms taps hooked, then merges into
  the validated `hessian.json` in `--out-dir`: it adds the files under `"files"`, `"rms_keys"` and
  `"rms_capture"` (provenance and gates), and keeps every other byte, rewriting the manifest once
  through the temp + rename writer with LF endings. Before any GPU work it refuses (also under
  `--dry-run`) unless the checkpoint's `config.json` sha256 matches the set's and this run's corpus
  matches the recorded one: every source's sha256, token counts, windows and file list. The
  repo-source windows come from the working tree, so a set captured at an older checkout needs
  `--code-rev <commit>`, which reads them from that commit's tree. `D:\models\r4dx\hessian-v1`
  matches at `--code-rev 34d381a`. The run also refuses a manifest that already has `rms_keys`
  unless `--force` is given, and then it removes the old rms entries before capturing. It refuses
  to merge if `hessian.json` changed during the capture. With `--rms-taps`, a full capture lists the
  rms files only if their own gates pass, and `--regate` keeps them and re-gates their diagonals with
  no exemption.

**Converter** (`hessian_store.hpp`, `rotation.hpp`, `main.cpp: add_linear`):

- `HessianStore` parses the optional `"rms_keys"` map. It refuses:
  - a map that is not an object, or a value that is not a string;
  - a file not listed under `"files"`;
  - a base with no `"keys"` entry;
  - the `"keys"` file itself, or any other base's `"keys"` file;
  - any file but the base's own rms twin (`L03.in.hess` -> `L03.in.rms.hess`). Every norm-fed tap
    has `K = hidden`, so only the name ties the rms file to the base's tap and layer;
  - a `K` different from that entry's file, or `rows` different when both are listed.

  `Factor(kRms)` also refuses an rms file with a zero diagonal. That is what a capture of the norm's
  output (or a weighted input) would give on a dead channel, and damping would hide it.

  `Factor(..., HessianSource::kRms)` factors the rms file. The file is part of the one-entry cache
  key, so `in_proj_qkv/z` share one factorization and so do `qg/k/v`. A base's `"keys"` file and its
  rms file never share one.
- `add_linear` decides once per linear: `use_rms = --ldlq selects it AND it is a rotated
  in-projection AND rms_keys has it`.
  - When `use_rms` is true, the Hessian is `TransformHessianRmsQ(H_rms) = Q^T H_rms Q`. There is no
    division and no norm weight. The transform id is `"<kind>:seed=<s>:in-rms"`, and planning checks
    the rms file's header and size.
  - Otherwise, a rotated in-projection falls back to the division path. Planning now reads the
    norm vector and refuses any `|1 + w| < 1e-3` before the header is written, not at that layer's
    emit. The message names the linear and the channel and says to run
    `hessian_capture.py --rms-only`.
- **Unrotated conversions never read an rms file** (`fold.kind` is `kNone`). They are byte-identical
  to before, whether or not `"rms_keys"` exists.
- `r4dx_convert_run.ldlq_rms_linears` (rotated containers only) lists the in-projections that used
  `H_rms`. The per-linear log line names the rms file.

**Tests.** `tests/convert/test_rms_hessian.cpp` (ctest `convert_rms_hessian`, CPU) covers:

- `TransformHessianRmsQ` against the dense closed-form `Q`, in full at 80 and 256 and on a sampled
  48 x 48 block at 5120;
- a dead channel on captured activations: the division path refuses; the rms path matches a direct
  capture on `rms(x) Q`, keeps `tr(W' H' W'^T) = tr(W H W^T)`, and gives `e_j Q` its energy
  `H_rms[j][j]`;
- every `rms_keys` refusal, and `Factor(kRms)` on a zero diagonal;
- `r4dx-convert` on a synthetic 2-layer checkpoint with a dead channel:
  - unrotated output matches a digest pinned from the pre-rms build, with and without `rms_keys`;
  - `q2ab` without `rms_keys` is refused during planning;
  - with `rms_keys`, `ldlq_rms_linears` is recorded, and the w4a16 bytes equal fold, then
    `Q^T H_rms Q`, then LDLQ;
  - a partial `rms_keys` mixes both paths in one run.

`tests/reference/test_hessian_rms.py` (ctest `reference_hessian_rms`, CPU, no checkpoint) drives a
tiny random 2-layer Qwen3_5 stack, with a dead channel, through the tool's own `run_capture`, gates,
`--rms-only` merge and `--regate`. It checks:

- `rms_weightless` is `Qwen3_5RMSNorm._norm(x.float())` bit for bit;
- each rms file equals an fp64 `E[rms(x)^T rms(x)]` of the recorded norm inputs;
- the gates pass on a consistent capture (worst 1.4e-3);
- a merge keeps every other manifest byte;
- nothing is merged after a different corpus, a zeroed rms diagonal, a rows mismatch, or a manifest
  that changed mid-run;
- `--regate` keeps `rms_keys` and gives the rms files no dead-channel exemption.

**GPU run (handed to the user).** With `$env:HIP_VISIBLE_DEVICES = '1'`, the merge into the existing
set is
`python tools\reference\hessian_capture.py --rms-only --code-rev 34d381a --out-dir D:\models\r4dx\hessian-v1`
(128 files, 6.25 GiB, one layer-major forward with nothing else hooked; `--dry-run` first checks the
corpus without the GPU). After that, `r4dx-convert --rotate q2ab --hessian-dir D:\models\r4dx\hessian-v1
--ldlq .` no longer needs the division path.

### 3.3 Corpus v2: more windows plus self-generated text

**Why.** On the held-out KL eval, Thai is the worst segment by 2x (section 7). Yet Thai is about 1%
of hessian-v1's 172k tokens: `kv_calib_corpus/thai_prose.txt`, 1,297 tokens. The held-out analysis
splits the remaining gap roughly in half:

- **Sampling.** The corpus is too small: rows/K is about 10 for K = 17408 (`mtp.mlp_mid` 9.9).
- **Domain shift.** The corpus text does not look like what the model is actually served.

Corpus v2 addresses both. It keeps v1 and adds two things:

- More of the same sources: WikiText-2 128 windows (the train file has about 2.4M tokens) and
  code 48 windows (the repo concatenation has about 1.3M).
- About 400k tokens of text **self-generated by our own quantized model** through r4dx-server, with
  no downloads. Categories: `thai_prose`, `english_prose`, `chat`, `code`, `multilingual`.

The generated text only has to be realistic input. The Hessians still come from running the bf16
checkpoint over it. The total is about 770k tokens, which puts rows/K at about 44.

**`--gen-file samples.jsonl`** (corpus source 4, `build_gen_corpus`). The input is one JSON object
per line: `id` `"<category>/<NNN>"`, `category`, `format` `raw|chat`, `enable_thinking`, the full
`messages` including every generated assistant turn and its `reasoning_content`, `text` (a raw
sample's final assistant content, stripped), `finish_reason`, the token counts, the sampling
settings, and `rejected`/`reject_reason`. `load_gen_samples` refuses any line that breaks this
contract.

Only samples with `rejected == false` are used, in id order:

- A **raw** sample is its `text`.
- A **chat** sample is rendered through the checkpoint's own `chat_template.jinja`, with
  `enable_thinking` as recorded and `add_generation_prompt=False`.

The template's behaviour, checked on every render by `check_rendered_chat`:

- Every assistant turn renders as `<think>\n{reasoning_content|trim}\n</think>\n\n{content|trim}`,
  because `preserve_thinking` is left unset. So the final turn's thought survives. With
  `preserve_thinking=false`, only the turns after the last user message would keep it; the final
  turn always does.
- A thinking-off turn gets the empty block `<think>\n\n</think>\n\n`. That is exactly the template's
  `enable_thinking=false` generation prompt, so the render is what the model was served plus what
  it generated.
- `enable_thinking=true` adds the default (`xhigh`) reasoning-effort sentence as a system turn.

**Packing.** Per category, the rendered samples are concatenated. A raw document is followed by a
blank line; chats go back to back, because each ends in `<|im_end|>\n`. The concatenation is
tokenized with `add_special_tokens=False`: the control tokens are in the text, and the checkpoint
adds no BOS. It is then cut into non-overlapping `--seq-len` windows. The short tail is dropped and
reported. `--gen-max-seqs N` keeps N evenly spread windows per category, a subset of the same grid.

**Provenance.** Each category gets a corpus source entry with:

- the file's path and sha256, and the chat template's sha256;
- the sample ids used, the rejected count per reason kind (the `<kind>` of the generator's
  `<kind>: <detail>`), and the formats;
- the concatenation's sha256 and token count;
- the window starts and the dropped tail.

So `--rms-only` accepts a later run only with the same jsonl, passed as `--gen-file`.

**Disjointness with `kl_corpus/`.** A hit is a 50-character whitespace-normalized run shared with
any `tools/reference/kl_corpus/*.txt`, outside that file's boilerplate lines. The boilerplate lines
are `#include` directives and Python import statements (`KL_BOILERPLATE_LINE`, including a whole
`from a import (` ... `)` block). The cpp and python KL excerpts are the headers of model.cpp and
layer_golden.py, and their sorted standard includes and imports match any C++ or Python text:
`\n#include <algorithm>\n#include <chrono>\n#include <` is 50 normalized characters on its own. The
rule skips 23 lines of `cpp_source.txt`, 21 of `python_source.txt` and none of the prose files.

- **Generated samples: refused.** The check runs before tokenizing, on each rendered sample. For a
  raw sample it also covers the prompt and every other message field: they are not calibrated on,
  but a prompt quoting KL text steers the output into the eval's domain. The refusal lists every
  offending sample with where the run is, the KL file and line; mark those samples rejected to
  proceed. The check is one hash lookup per character, about 0.2 s for 1.5M characters.
- **Sources 1-3: checked, recorded, not refused.** Every token window is decoded back to its text
  and checked the same way. A hit is printed as a WARNING, naming the repo file for a code window,
  and listed under that source's `kl_disjointness.overlaps`; the CAVEAT in `hessian.json` says so.
  Refusing is not workable here. model.cpp and layer_golden.py are left out of the code
  concatenation, but 25 of the 277 other files share runs with them: model.h and three more repeat a
  model.cpp comment, `container.cpp` repeats a block of its code, ten files repeat the venv usage
  line of layer_golden.py's docstring, and the rest share single lines of idiomatic code. Whether a
  window lands on one depends on `--code-seqs` and on the
  commit. On the working tree, `--code-seqs 48` puts `code/41` on that usage line, and 17 of 21
  counts between 16 and 64 hit at least once. hessian-v1's own `code/04` (at `34d381a`) holds the
  model.cpp comment, so a refusal would also break `--rms-only` on hessian-v1. calib.txt,
  `kv_calib_corpus/` and the first 3M characters of WikiText-2 have no hit.

Every source entry records the check as `kl_disjointness`: the KL files' sha256, the boilerplate
line counts, what was checked, and `overlaps`/`ok`. `--rms-only` ignores that field when it
compares corpora. Lines that can carry Thai go through `say`, and `main()` switches stdout to
`backslashreplace`. On Windows a pipe or a redirected file gets cp1252 with `strict` errors, so
without this the refusal would die in a `UnicodeEncodeError`.

Without `--gen-file`, the corpus tokens are identical to before; the source entries only gain
`kl_disjointness`. At `--code-rev 34d381a`, the `--dry-run` still gives hessian-v1's 86 sequences
and 172,156 tokens, with the same token ids, and `--rms-only --dry-run` still reports "corpus
matches". It now also warns about `code/04`.

**GPU run (handed to the user).** Do the `--dry-run` first with the same flags. Then, with
`$env:HIP_VISIBLE_DEVICES = '1'`:

```
python tools\reference\hessian_capture.py --out-dir D:\models\r4dx\hessian-v2 --rms-taps `
    --wikitext-seqs 128 --code-seqs 48 --gen-file D:\models\r4dx\corpus-v2\samples.jsonl
```

This writes 389 files and about 54 GiB. Expect roughly 4.5x v1's 15 minutes. The hidden states
(about 7.3 GiB) exceed `--hidden-device auto`'s 6 GiB, so they live on the host. The code windows
come from the working tree at capture time, so a later `--rms-only` needs `--code-rev <that commit>`.
Expect the dry-run to print a disjointness WARNING for a code window or two (above); a refusal
names generated samples only.

**Tests.** `tests/reference/test_hessian_corpus.py` (ctest `reference_hessian_corpus`, CPU) runs on
the fixture `tests/reference/fixtures/hessian_gen_samples.jsonl`: raw samples, a chat without
thinking, a two-turn thinking chat, and one rejected sample. It checks:

- the contract refusals;
- the renders against the template, and their token ids;
- packing and the dropped tail;
- `--gen-max-seqs`;
- determinism under reordering;
- provenance and `corpus_mismatches`;
- that the gate trips on KL excerpts read at runtime (only in a temp dir): in a raw text, in a raw
  sample's prompt, and in a chat's user turn. It does not trip at 49 characters, nor on the generic
  include/import headers and the KL code files' own include/import lines. It lists all 10 of 10
  overlapping samples, and survives a strict cp1252 stdout;
- the boilerplate grammar, and the runs and line numbers `kl_gate_segments` produces;
- that a calib window quoting KL text is recorded under its own source and not refused, and that a
  code window's overlap names the file (the file list and blobs are stubbed).

## 4. Q2b -- online Hadamard on the down / o / out_proj inputs

`W' = W Hb` on the K side with `Hb` blockwise normalized Hadamard (random signs from the same seed),
input rotated online by `Hb^T = Hb`:

| linear | K | block | where the rotation runs |
|---|---|---|---|
| `mlp.down` | 17408 (8704 per TP rank) | 512 | fused into `r4dx_silu_mul_bf16` (new `..._hadamard` entry) |
| `attn.o` | 6144 (24 heads x 256) | 256 | fused into `r4dx_model_attn_gate_mul_bf16` |
| `gdn.out_proj` | 6144 (48 heads x 128) | 128 | a separate in-place kernel after the gated norm (the decode norm lives inside libr4d's recurrent kernel) |

Cost gate G7 decides whether `gdn.out_proj`'s extra launch stays.

## 5. Q3 -- per-tensor w4a16 group

- libr4d: `R4D_GEMM_W4_GROUP` becomes a template parameter; one new entry point
  `r4d_gemm_w4a16_nt_m64_g(int group, ...)` (32/64/128 instantiated) plus
  `r4d_gemm_w4a16_nt_m64_has_group(int)`, and the existing entry (build default) unchanged for
  compatibility.
- Container: `__metadata__.quant.w4a16.groups` maps base name -> group when not all equal; the
  loader dispatches per `QuantLinear`; the tuning table gains a group column; the TP slicer
  already takes the group per tensor (`ShardLoader`'s `w4a16_group_` becomes per call).
- Allocation: the Milestone 11 method, automated -- measure per (class, depth half) the KL
  recovered by g32 / g64 / g128 (one class at a time, rung 4), then fill the byte budget greedily by
  nats per GiB and stop at the cliff. The same sweep also measures bf16 keeps of a few sensitive
  linear sets and un-keeps of the recipe's bf16 attn.k/v (5.3), so every option is one precision
  (g32 / g64 / g128 / bf16) for one set of linears, and sets that overlap exclude each other.

### 5.1 Q3 implementation note (runtime)

The on-disk format is docs/container-format.md, w4a16, "Per-tensor groups": `quant.w4a16.group`
stays the container's default, the optional `quant.w4a16.groups` lists only the linears at another
group, and such a linear's scales are `<base>.w4a16.wsz.g<g>` instead of `<base>.w4a16.wsz`. The
converter side is `r4dx-convert --w4a16-group-rule` (README). This note covers the runtime half.

**Off by default.** A container without the map takes the pre-Q3 path everywhere. The parse returns
an empty map. Every scale tensor keeps its bare name. Every `QuantLinear::w4a16_group` stays 0, which
means "this build's default". `ApplyLinear` then calls the historical `r4d_gemm_w4a16_nt_m64` entry
with the tuning `PickTuning` has always returned, and the group check runs up front exactly as
before. The kernel code for that entry is unchanged: the libr4d change diffs the 16 default
instantiations' ISA as identical at both build defaults. The runtime has no flag for this: the
container decides.

**Loader** (`src/model/w4a16_group_meta.h`, `src/model/container.cpp`, `src/model/quant_linear.h`):

- `w4a16_group_meta.h: ParseW4a16Groups` is HIP-free and header-only, like `rotation_meta.h`. It
  runs on every load, TP=1 and TP alike, straight after the metadata read. It returns the default
  group (128 when the `quant` block predates it, the historical rule) and the map. It throws, naming
  the path, when:
  - `groups` is not an object, or is present without `group`;
  - a key is empty;
  - a value is not an integer, not 32/64/128, or equal to the default.

  `W4a16Groups::GroupFor`/`WszName`/`QuantLinearGroup` are the only places that turn a base name into
  a group, a scale-tensor name or a `QuantLinear::w4a16_group`.
- `container.cpp: CheckQuantGroups` still fires only when the load selects w4a16 for the body, the
  lm head or the MTP head. It now returns `W4a16LoadGroups`, which every w4a16 read goes through.
  - No map: `CheckW4a16Group(default)` up front, unchanged.
  - A map: every mapped group must pass `r4d_gemm_w4a16_nt_m64_has_group` up front
    (`CheckW4a16MappedGroup`, a `W4a16GroupMismatch`). The default-equality check then applies only
    to linears without a map entry. It runs when the first such linear is read as w4a16
    (`W4a16LoadGroups::Resolve`) and throws the same exception type. In practice every real
    container has unmapped w4a16 linears, so a binary whose default differs from the container's
    still refuses it.
- `CheckW4a16GroupTensors` runs once per load, before any upload, for any layout. Each mapped base
  must carry `.w4a16.wq` and `.w4a16.wsz.g<g>`, and must NOT carry a bare `.w4a16.wsz`. This check
  rejects a map entry for an unknown base.
- `LoadQuantLinear` (TP=1) reads the linear's own scale name and sets `w4a16_group`. It checks
  `wq` = N·K/2 bytes, the scales = N·(K/g)·4 bytes, and `K % g`, `K % 64`, `N % 16`. The TP loader
  already checked every part's size. At TP=1 this check is new for every w4a16 linear, but a
  container the converter wrote always passes it.
- `LoadQuantLinearWithFallback` and `ShardLoader::Linear`: a mapped base requested as w4a16 never
  falls back to `.bf16.w` or the bare name. It throws instead. Unmapped bases keep the
  requested -> bf16 -> bare chain unchanged.
- TP: `ShardLoader` holds the `W4a16LoadGroups` instead of one int. It cuts each linear's wsz at the
  linear's own group (`tp::PartShape::group`), under the linear's own name. `tp_shard.cpp` needed
  no change, because `PlanRows`/`PlanCols` were already generic in the group. A rank's K range is
  whole 64-K blocks (wq's rule), so it is whole groups at 32, 64 and 128. mlp.down's per-rank
  8704 = 136 × 64 = 272 × 32.
- The DFlash2 drafter refuses a container that carries a non-empty map
  (`dflash_draft_weights.h`). The converter never writes one for a drafter.
- One stderr line announces a map when the load selects w4a16 (TP: rank 0 only).

**Dispatch and tuning** (`src/model/linear.{h,cpp}`, `src/core/include/r4dx/core/r4d.hpp`):

- `ApplyLinear`: `EffectiveW4a16Group(w.w4a16_group)` (0 becomes `r4d_gemm_w4a16_nt_m64_group()`).
  If the result equals the build default, it calls `core::r4d::GemmW4a16NtM64`, the historical
  entry. Any other group calls the new `GemmW4a16NtM64G(group, ...)`, which wraps
  `r4d_gemm_w4a16_nt_m64_g`. The default path adds one int compare per GEMM chunk.
- `GemmTuningRow` gains a trailing `int group = 0`, so generated `.inc` rows still compile
  unchanged. For w4a16 the group is part of the key: a row's group is `row.group`, with 0 meaning
  the build default. The request's group is `EffectiveW4a16Group`.
  - At the default group every existing row matches. The legality test becomes
    `K % (SK * max(g, 64))`, which equals the old `K % (SK * g)` for every default a build
    accepts, so the default path picks exactly what it always did.
    `test_pick_tuning` checks this against a verbatim copy of the pre-Q3 resolution, for every row,
    both tables and M = 1..64.
  - At a non-default group no row matches yet, because `tune_gemm.py` writes group 0. The pick
    falls back to `FallbackTuning` (WV4/SK4/MB1/NPW1). That tuning is legal at 32, 64 and 128 for
    every K this model has, TP ranks included, since `K % 512` covers `4·max(g, 64)`. It gets the
    same kRowTile treatment (M <= 16 shares M=1's SK, NT=0 on M = 2..16). The PickTuning cache key
    carries the effective group in bits 53 and up.
- The kernel's K rule for a split is `K % (SK * max(group, 64))`. A split must start on a 64-K
  packed block, which a group of 32 does not guarantee. `BestRow`, `FallbackTuning` and
  `test_pick_tuning`'s `Launchable` all use this form.

**Old binaries refuse a mixed-group container.** This is reasoned from `main` at `e18a3a9`
(`src/model/container.cpp`). Its `CheckQuantGroups` reads only `quant.w4a16.group`, which is the
unchanged default, so the load proceeds. For a mapped linear, `LoadQuantLinearWithFallback` calls
`HasLayout(kW4a16)`. That call needs the bare `<base>.w4a16.wsz`, which the container does not
have. The fallback chain then tries `<base>.bf16.w`, then the bare `<base>`, and throws "no tensor
found for '<base>' in any known on-disk form". The TP `ShardLoader::Linear` on `main` runs the same
chain and throws the same error. The drafter never meets a map.

That refusal depends on the linear having no `.bf16.w`. With a bf16 companion, the old binary
would silently load the bf16 copy: correct numbers, the wrong layout, about 3.5× the bytes, and only a
fallback count in the log. So `r4dx-convert` refuses a non-default group on any linear that keeps
`.bf16.w`. Use `--no-bf16`, plus an `--lm-head` spec without bf16. The v6 recipe already does both.
Loads with `--layout mxfp4` or `w4a8` on an old binary are correct, because the map changes only
w4a16 tensors. `r4dx_format_version` is not a guard, because no reader checks it.

**Costs and consequences.**

- Speed: every mapped linear runs untuned (FallbackTuning) until `tools/profile/tune_gemm.py` sweeps
  `r4d_gemm_w4a16_nt_m64_g` at that group and writes rows with the group column. Its parser and
  writer do not yet know the column. The KL sweep is unaffected, but G8's decode number for a
  mixed container needs those rows.
- Determinism: a linear's arithmetic depends on its group and SK. kRowTile holds per group, so the
  verify-row exactness of f7d4927 holds on a mixed container too.

**Tests.** These CPU tests always run:

- `test_w4a16_group_meta` checks:
  - the parse and 16 refusal cases;
  - the scale-tensor names;
  - a round trip through the converter's own `W4a16GroupRules`;
  - every real linear's TP=2 wsz slice at 32/64/128.
- `test_tp_shard`'s variants gain w4a16 g32, byte-exact against the converter's packers, with a
  pinned rank-1 mlp.down g32 offset.
- `test_pick_tuning` covers groups 0/32/64/128 for launchability, the pre-Q3 pick at the default,
  the fallback at other groups, the row-tile SK, and `has_group`.

`test_tp_loader`, a GPU test, now resolves each linear's wsz name and group through the map. The
GPU side (a mixed container loaded and decoded, TP=1 and TP=2, plus rung-4 KL per candidate) is
handed to the user.

### 5.2 Reusing a baseline (`--reuse-tensors-from`)

The Q3 sweep converts a baseline and 41 candidates, each the baseline plus ONE change -- a
`--w4a16-group-rule`, or a `--keep-bf16` that keeps more or fewer linears in bf16 (5.3) -- at ~24 min
each as full conversions.
`r4dx-convert --reuse-tensors-from <baseline.r4dx>` writes exactly the container a full run with the
same flags writes, but copies from the baseline every tensor that cannot depend on the difference
(`src/convert/main.cpp`: `BuildReuseGuard`, `OpenReuseBaseline`, `CheckReuseBaseline`, `PlanReuse`;
`reuse_guard.hpp`).

**Recomputed:** every linear whose RESOLVED layout set differs between the two runs, read from each
run's `reuse_guard.linears` (below): `"w4a16.g64"`, `"w4a16.g32"`, `"bf16"`, ... per linear, after
the rules and `--keep-bf16` (`LayoutSetId`, `linear_layouts.hpp`). That covers a rule in this run, a
rule only the BASELINE had, two different groups, a linear either run keeps in bf16 and the other
quantizes, and any mix of these; neither flag's regex text is compared, so two spellings of one keep
set recompute nothing. All of that linear's layouts are recomputed, with everything its job writes
(the MTP draft head's `vocab_ids` too). **Copied:** everything else. Each linear is quantized from its
own weight and its own Hessian or imatrix vector, and nothing reads another linear's quantized
output. HessianStore's factor cache is only a cache: a reuse run may factor a shared tap afresh, and
`convert_reuse` (d) and (e-iv) check the bytes do not care.

What `--keep-bf16` touches beyond a kept linear's own tensors is either recomputed or independent of
it (the audit behind 5.3): `keep_bf16`, `keep_bf16_linears`, `keep_bf16_extra_bytes`,
`ldlq_linears`, `ldlq_rms_linears`, `quant.w4a16.groups` and `w4a16_group_extra_bytes` all come from
the planning pass, which a reuse run executes in full; the `.hess` files the guard hashes follow
`--ldlq` alone (`HessianStore::FilesFor`), so un-keeping a linear reads no unhashed file; the
kv-calib descales, the zeroed norms and the `rotation.*` tensors do not consult `--keep-bf16`; a kept
linear is folded (`W diag(1+w) Q`, `Q^T W Hb`) exactly like a quantized one, then rounded to bf16
once.

The baseline's record is not trusted on its own. Before anything is copied: the record must name
exactly this run's linears, each a canonical layout set; the baseline's `quant.w4a16.groups` and
`keep_bf16_linears` must agree with it; a recomputed linear's baseline tensors must be exactly the
ones its record names; every copied tensor must exist in the baseline under the same name with dtype
U8, the same shape and the same size; and the baseline must hold nothing else. The copy streams from
the baseline's mapping in 64 MiB writes.

**The guard** is `__metadata__.r4dx_convert_run.reuse_guard`. `--record-reuse-guard` writes it, and
so does every reuse run. A baseline without a guard (any older converter's container), or with an
unfinished one, is refused from its header alone, before anything is hashed. Otherwise a reuse is
refused, before the first shard is read and naming each field, unless the baseline's guard equals
this run's in every field:

| field | content |
|---|---|
| `version` | the guard format (3: `args.keep_bf16` gone, `linears` added) |
| `converter.exe_sha256` | the running `r4dx-convert.exe` (r4d_core is linked statically) |
| `converter.runtime_dlls` | ucrtbase / msvcp140 / vcruntime140 as loaded (`rotation.mix5` uses ucrt's log/cos/sin) |
| `converter.cpu` | vendor, brand, family/model/stepping, FMA3/AVX/AVX2/AVX-512F, XCR0, `_get_FMA3_enable()`: ucrt picks its FMA3 or SSE2 transcendentals at run time, and they can differ in the last bit |
| `converter.{w4a16,w4a8,mxfp4}_group`, `avx512` | build constants; the LDLQ microkernel path |
| `checkpoint` | sha256 of `config.json` and `model.safetensors.index.json`; length + sha256 of every shard the index names (whole files). The path is not compared. |
| `args` | resolved `layers`, `vision`, `mtp`, the body and lm_head layout sets after `--no-bf16`, `quant`, `ldlq`, `ldlq_damp`, `rotate`, `rotation_seed` |
| `inputs` | sha256 of `--imatrix`, `--kv-calib`, `--draft-vocab-ids` (when `--mtp` is on) and `hessian.json`; `hessian_files`: length + sha256 of every `.hess` file the `--ldlq` regex can make the run read (the selected bases' `keys` and `rms_keys` files) |
| `linears` | not compared as a whole: every linear's resolved layout set (`{"text.layers.0.mlp.down": "w4a16.g64", "text.layers.3.attn.k": "bf16", ...}`); `PlanReuse` compares it linear by linear (above) |
| `emit_complete`, `data_sha256` | not compared: the completion fields (below) |

`hessian.json` alone does not identify the Hessians. It pins each file's K, rows and header trace
(`CheckFile` holds the header to them, the trace exactly, so two same-K files swapped under one
manifest are refused even by a full run), and `ReadHessFile` ties the trace to the diagonal. But
nothing ties the rest of a payload to the manifest, so an in-place edit that keeps the diagonal
passes every check. A reuse run never reads the files of the linears it copies, so only
`hessian_files` can say that the baseline rounded against the same Hessians. The checkpoint's
shards and the `.hess` files are hashed in one parallel pass (about 54 GiB for hessian-v1 with
`--ldlq ".*"`, on NVMe).

Not compared: the group rules and the `--keep-bf16` regex (both resolved into `linears`), `--output`,
`--threads` (the bytes do not depend on it, gate G1(c)), and input file paths. There is deliberately
no `--reuse-allow-binary-mismatch`. Any rebuild can change
what a quantizer writes. A container mixing two converters' tensors would measure as a group-rule
effect, silently. The price of refusing is one baseline conversion. Also refused: a baseline without
a guard, `--output` equal to the baseline, a header that is not JSON, and a quant block from
another build.

**Completion and data digest.** ContainerWriter pre-sizes the file, so an interrupted conversion
looks valid: a whole header over zero-filled tensors. So the guard's two completion fields are
written as placeholders (`emit_complete` 0, `data_sha256` 64 zeros). After `writer.Finish()` the
converter syncs the file to disk (`_commit`), reads every tensor back in parallel, and records
`data_sha256`. That is the sha256 over `"<name>\n<sha256 of the tensor's bytes>\n"`, names in byte
order. Then `emit_complete` flips to 1. Each patch is durable: the data reaches the disk before the
patch is written, and the patch before the call returns. A reuse run refuses a baseline whose
marker is not 1 or whose digest is still the placeholder. It then hashes the baseline's every
tensor, in parallel and straight from its mapping, and refuses it unless they give `data_sha256`.
That catches a completed baseline that was later partly overwritten, or copied by a tool that died
half way (full length, header intact, a zero tail). The mapping is opened share-read only, so the
baseline cannot change between that check and the copy. The copy pass then checks that every copied
tensor reads back from the output with the baseline's digest.

**What differs from a full run.** The data section is identical byte for byte, and so is the
recorded `data_sha256`. Against a full run with `--record-reuse-guard`, the header text differs
only by `r4dx_convert_run.reused_from` (path, the baseline's header sha256 and `data_sha256`,
tensors / bytes copied and recomputed, `linears_recomputed`), plus `threads` if `--threads`
differed. Against a full run without the flag, it also has `reuse_guard`. Without either flag no
hashing happens and the header is unchanged. The guard hashes the whole checkpoint, one shard per
worker at ~350 MB/s each: 30 s for the 27B checkpoint (55.6 GB, 18 shards, 32 threads), measured
2026-09-26. The `.hess` files, the baseline's data check and the output's read-back digest (18.5 GB
each, one tensor per worker; the bf16 `embed_tokens`, ~2.5 GB, is the longest single item) have
not been timed on the 27B yet. Nor has the 18.5 GB copy.

**Sweep:** `tools/quant2/group_sweep.ps1 -Convert` converts the baseline with `--record-reuse-guard`,
keeps it, and passes `--reuse-tensors-from` to every candidate. `<name>.meta.json` records
`reused_from`, the container's `data_sha256` and its guard identity (`guard_json`: the guard without
the completion fields and without `linears`, the one field the candidates change). A measured
baseline is never replaced silently. If its container is gone,
it is converted again, and the new container must have the recorded `data_sha256` and `guard_json`.
Otherwise the sweep stops, because `kl_baseline.json` and every candidate measured against it are
stale; start a new `-OutDir`. A measured baseline from an older sweep, with no digest recorded,
cannot be checked, so continue it with `-NoReuse`. Since guard v3, `guard_json` no longer carries the
recipe's `--keep-bf16` (a keep candidate changes it), so `-Kl` leaves out a candidate whose
`guard_json` differs from the baseline's, whose `keep_bf16_linears` is not the baseline's plus (keep) /
minus (unkeep) / unchanged by (group) its own set, or whose `baseline_data_sha256` (the baseline it
was planned and verified against) is not the measured baseline's `data_sha256`. A measured baseline
also fixes the recipe's `--keep-bf16` for its `-OutDir`: a run whose `-Recipe`/`-ExtraArgs` give
another is refused before any work, and `candidates.json` carries the baseline's own `convert_args`
as the recipe. `-ReuseFrom <container>` uses another source; `-NoReuse` is the old full-conversion
path.

**Test:** `convert_reuse`, CPU, ~45 s. It runs a synthetic 2-shard checkpoint through the sweep's
recipe (imatrix, kv-calib, MTP, draft head, vision) and through `--rotate q2ab --ldlq ".*"` with rms
Hessians. It checks:
- reuse(R) == full(R), and the reverse, `data_sha256` included;
- chained reuse, a group change on both sides, and a default-group rule;
- a moved checkpoint and another `--threads` are accepted;
- every flag, input file, checkpoint file and binary difference is refused, and so is each of the
  64 guard leaves (the 17 `linears` entries included) mutated in a copy of the baseline;
- every inconsistent or unfinished baseline is refused. That includes one whose data no longer
  matches its digest (half zeroed at full length, or one bit flipped), and one whose per-linear
  record is missing, names a linear this run does not write, misses one, disagrees with its group map
  or keep list, or names tensors the baseline does not have;
- a `.hess` payload edited in place under an unchanged `hessian.json` is refused. So are two
  swapped same-K files, even by a full run;
- (e) `--keep-bf16` differing: a keep added (layer 0 `mlp.down`, one linear recomputed, the
  `keep_bf16_extra_bytes` delta exactly `2NK - w4a16(g64)`), a keep removed (`attn.v`), the reverse,
  another spelling of the same keep set (nothing recomputed), a keep together with a group rule on
  the same class (keep wins; from the no-rule baseline and from a baseline with rules), each ==
  its full run; a keep change together with any other difference is still refused; and on the
  q2ab + LDLQ fixture, layer 0 `mlp.down` (down Hadamard), layer 1 `attn.o` (o Hadamard) and
  `attn.k` (rms tap shared with `attn.qg`/`attn.v`) kept: their bf16 bytes are the folded weights,
  reuse == full both ways, and un-keeping them LDLQs exactly those three (`attn.k`'s factor fresh
  where the full run took it from the cache).

**Measured on the 27B (2026-09-26, `D:\models\r4dx\q3-reuse-test`, recipe `run_variants.ps1`'s base +
`--rotate q2ab --hessian-dir hessian-v1 --ldlq .`, while corpus v2 generation ran on the GPU):** the
guarded no-rule baseline took 25.8 min. Candidate A (`mlp.down` in layers 0-31 at group 32)
took **8.7 min** by reuse (64 tensors / 1.78 GB recomputed, 1423 / 18.2 GB copied) against **23.4 min**
for the full conversion. `compare_containers.py`: the tensor directory is identical, all 1487 data
ranges are identical, both files' `data_sha256` is `3408ff4d...` and recomputes; `__metadata__`
differs only under `r4dx_convert_run.reused_from`. The pre-feature `qwen38-27b-q2ab_ldlq.r4dx` was
refused as a baseline (no guard). Most of a reuse run is the recomputed linears' own LDLQ
factorizations (an `mlp.down` factor is ~12.6 s, 32 of them), so a candidate on a smaller class is
cheaper still.

### 5.3 Sensitive-tensor candidates (bf16 keep)

llama.cpp's `Q4_K_M` keeps `attn_v` and `ffn_down` at a higher precision in half the layers. r4dx has
one higher precision, bf16 (`--keep-bf16`): 2 bytes per weight against w4a16 g64's 0.5625, so keeping
a linear adds 1.4375·N·K bytes, 3.56x its w4a16 size. v6 bought bf16 `attn.k`/`attn.v` (+0.2246 GiB)
at Milestone 11's 0.0168 nats/GiB (`docs/validation.md`, v5 recipe). On the current recipe that price
no longer holds:

| container (section 7) | attn.k / attn.v | mean KL | weights |
|---|---|--:|--:|
| q2ab_ldlq | bf16 | 0.02261 | -- |
| q2ab_nokv | w4a16 g64 | 0.02333 | -0.2246 GiB |

That is +0.00072 nats for 0.2246 GiB, **0.0032 nats/GiB**, a fifth of Milestone 11's figure: LDLQ and
the rotation removed most of the error bf16 was buying back, so the Milestone 11 per-class ranking is
stale for every class, not just k/v. The sweep therefore measures bf16 keeps on the current recipe as
candidates in the same currency as the groups, and prices the recipe's own k/v keep as two savers.

The current recipe is q2ab_ldlq (section 7). `group_sweep.ps1`'s default `-Recipe` alone is the
unrotated v6 recipe, so the Q3 round runs:

```powershell
.\tools\quant2\group_sweep.ps1 -Convert -Kl `
    -ExtraArgs '--rotate','q2ab','--hessian-dir','D:\models\r4dx\hessian-v1','--ldlq','.'
```

Every later run on that `-OutDir` (a bare `-Kl` re-collect included) passes the same `-ExtraArgs`.
The startup line prints the recipe's `--keep-bf16`, `--rotate` and `--ldlq`.

**Candidate kinds** (`tools/quant2/group_sweep.ps1`; `-CandidateFile` rows
`{"name", "kind", "bases_regex", "group"}`, a row without `kind` is a group candidate):

- `group`: the set moves to g32 / g128 (`--w4a16-group-rule "<bases_regex>=<g>"`), as before.
- `keep`: the set is written in bf16. `--keep-bf16` is single-valued (r4dx-convert keeps the last
  one), so a second flag would silently REPLACE the recipe's k/v term and confound the candidate
  with "k/v quantized". The recipe's `--keep-bf16` is replaced by ONE merged regex,
  `(?:<recipe>)|(?:<bases_regex>)`.
- `unkeep`: the set, which the recipe keeps, is quantized like every other linear (LDLQ included):
  `^(?![\s\S]*?(?:<bases_regex>))[\s\S]*?(?:<recipe>)`, which matches iff the recipe's regex matches
  and the candidate's does not, anchored or not.

Every candidate is verified against the converter's output, not its regex text. Before converting,
its intended set is the baseline's linears that `bases_regex` matches, each in the state the kind
needs (w4a16 for group / keep, kept for unkeep). After converting, the linears whose layout differs
from the baseline's (read from the two tensor directories) must be exactly that set, at the
candidate's precision; the converter's `keep_bf16_linears` must be the baseline's plus / minus it;
and a reuse run's `reused_from.linears_recomputed` must be the same set. `<name>.meta.json` records
`kind`, `precision`, the set (`linears`) and `extra_bytes`: what a `--layout w4a16` load reads (wq +
wsz, or bf16.w), candidate minus baseline, cross-checked against the converter's
`w4a16_group_extra_bytes` / `keep_bf16_extra_bytes` delta (the baseline already carries +241,172,480 B
for its 32 k/v linears). Reuse (5.2) recomputes only the changed linears; a kept linear is folded and
rounded to bf16 without LDLQ, so a keep candidate converts faster than a group candidate of the same
class. `-ListCandidates` prints the list, with each candidate's resolved linear count and byte delta
once the baseline is converted.

**Allocation** (`tools/quant2/alloc_groups.py`): every option is one precision for one linear set.
Two candidates whose recorded linear SETS overlap are mutually exclusive -- `mlp.down.L0-3.bf16` and
`mlp.down.L0-31.g32` were each measured with the other's linears at the baseline's precision, so
their deltas do not add (and keep wins over a rule in the converter). An overlapping saver never
funds a spender. Unkeep savers are ordinary savers and can fund a keep or a g32 spender.

The greedy fill alone settles an overlap by ORDER: free savers are taken first, then spenders by
nats/GiB, and the first pick locks its linears. Every default keep lies inside a g32 and a g128
candidate of its class half, so a free g128 half, or a g32 half slightly ahead by nats/GiB, would
lock out a keep that recovers several times more. Two steps follow the fill:

- **Exchange pass.** Each candidate left out because it overlaps a pick is priced as a swap. The
  picks it overlaps go out, it goes in, the shortfall is funded by the cheapest disjoint savers
  (judged by total KL), and the fill re-runs on what that frees. The best improving swap is made, and
  the pass repeats until no swap improves.
- **Exhaustive step.** The first-order optimum is computed exactly, cluster by overlap cluster, on
  the bytes/dKL Pareto frontier. The default list's clusters have at most three candidates, and the
  step takes milliseconds for the 41 candidates. It replaces the picks only when strictly better.
  On seeded random inputs shaped like the default list, the fill alone reaches the optimum in
  151/300, the fill plus exchange pass in 275/300, and all three steps in 300/300
  (`--self-test`).

Candidates left out for overlap are named with their role, free savers included.

The flags printed for the picks carry one `--keep-bf16` that replaces the recipe's (recipe + keeps -
unkeeps), with the linear set it must resolve to. The recipe is the measured baseline's
`convert_args`. The allocator checks, with Python's `re`, that the regex matches exactly that set
among every linear the file names: the baseline's w4a16 and kept linears and every candidate's set.
If a recipe does not match the baseline's, the allocator exits 2 at that point instead of producing
a combined container that differs from the prediction. After converting, check the combined
container's `keep_bf16_linears` too.

**Runtime support** (all ten classes run as bf16 inside a w4a16 container; M11 = measured kept in
Milestone 11, unrotated):

| class | TP=1 | TP=2 | q2ab-rotated |
|---|---|---|---|
| `gdn.in_proj_qkv` | `LoadQuantLinearWithFallback` loads `.bf16.w`, `ApplyLinear`'s kBf16 case; M11 | `ShardLoader::Linear` slices the bf16 form, `Rows{KeyDim,KeyDim,ValueDim}`; untuned (FallbackTuning) | in-projection: `W diag(1+w) Q` folded in fp32, rounded to bf16 once (the path the always-bf16 `gdn.in_proj_a/b` take) |
| `gdn.in_proj_z` | same; M11 | `Rows{ValueDim}`; untuned | in-projection fold |
| `gdn.out_proj` | same; M11 | `Cols(ValueDim)`; untuned | `Q^T W Hb` folded before the bf16 emit; the runtime's Hadamard keys on its signs, not the weight's layout. **Not yet measured.** |
| `attn.qg` | same; M11 | `Rows{2*attn_out}`; its rank shape has a bf16 row | in-projection fold |
| `attn.k`, `attn.v` | bf16 in the recipe | tuned bf16 rows at 512x5120 | measured: q2ab_ldlq, TP=2 emulate smoke 162/0 (G6) |
| `attn.o` | same; M11 | `Cols(attn_out)`; untuned | `Q^T W Hb`; the gate-multiply Hadamard keys on its signs. **Not yet measured.** |
| `mlp.gate_up` | same; M11 (two depth halves) | `Rows{I,I}`; untuned | in-projection fold (post_attention_layernorm) |
| `mlp.down` | same; M11 | `Cols(intermediate)`; untuned | `Q^T W Hb`; the Hadamard fused into silu_mul keys on its signs. **Not yet measured.** |
| `lm_head` | M11 (+1.739 GiB, 0.0033 nats/GiB) | vocab split; untuned | never folded (outside the rotated stack) |

**Byte prices** (w4a16 g64 = N·K/2 + N·K/16 bytes, bf16 = 2·N·K):

| class | [N, K] | linears | w4a16 g64 / linear | bf16 / linear | keep delta / linear | whole class kept |
|---|---|--:|--:|--:|--:|--:|
| `gdn.in_proj_qkv` | [10240, 5120] | 48 | 29,491,200 | 104,857,600 | +75,366,400 | +3.3691 GiB |
| `gdn.in_proj_z` | [6144, 5120] | 48 | 17,694,720 | 62,914,560 | +45,219,840 | +2.0215 GiB |
| `gdn.out_proj` | [5120, 6144] | 48 | 17,694,720 | 62,914,560 | +45,219,840 | +2.0215 GiB |
| `attn.qg` | [12288, 5120] | 16 | 35,389,440 | 125,829,120 | +90,439,680 | +1.3477 GiB |
| `attn.k`, `attn.v` | [1024, 5120] | 16 + 16 | 2,949,120 | 10,485,760 | +7,536,640 | +0.2246 GiB (both) |
| `attn.o` | [5120, 6144] | 16 | 17,694,720 | 62,914,560 | +45,219,840 | +0.6738 GiB |
| `mlp.gate_up` | [34816, 5120] | 64 | 100,270,080 | 356,515,840 | +256,245,760 | +15.2734 GiB |
| `mlp.down` | [5120, 17408] | 64 | 50,135,040 | 178,257,920 | +128,122,880 | +7.6367 GiB |
| `lm_head` | [248320, 5120] | 1 | 715,161,600 | 2,542,796,800 | +1,827,635,200 | +1.7021 GiB |

**Default keep / unkeep candidates** (after the 30 group candidates; GiB against the baseline):

| name | kind | layers | linears | extra GiB |
|---|---|---|--:|--:|
| `mlp.down.L0-3.bf16` | keep | 0-3 | 4 | +0.4773 |
| `mlp.down.L60-63.bf16` | keep | 60-63 | 4 | +0.4773 |
| `mlp.gate_up.L62-63.bf16` | keep | 62, 63 | 2 | +0.4773 |
| `gdn.in_proj_qkv.L0-7.bf16` | keep | 0-7 (GDN) | 6 | +0.4211 |
| `gdn.in_proj_qkv.L56-63.bf16` | keep | 56-63 (GDN) | 6 | +0.4211 |
| `gdn.out_proj.L0-15.bf16` | keep | 0-15 (GDN) | 12 | +0.5054 |
| `gdn.out_proj.L48-63.bf16` | keep | 48-63 (GDN) | 12 | +0.5054 |
| `attn.o.L0-31.bf16` | keep | 3, 7, ..., 31 | 8 | +0.3369 |
| `attn.o.L32-63.bf16` | keep | 35, 39, ..., 63 | 8 | +0.3369 |
| `attn.kv.L0-31.g64` | unkeep | 3, 7, ..., 31 | 16 | -0.1123 |
| `attn.kv.L32-63.g64` | unkeep | 35, 39, ..., 63 | 16 | -0.1123 |

Quantizing all of k/v measured +0.00072 nats, so each half should move about 0.0004; splitting k from
v is probably below what rung 4 resolves. `lm_head` bf16 (+1.7021 GiB, ~3x a candidate) is left out:
`lm_head.g32` (+0.074 GiB) is its affordable probe, and a keep regex for it must be `^lm_head$`
(`lm_head$` also keeps `mtp.draft_head.lm_head`). Lower-prior sets for a `-CandidateFile`:
`gdn.in_proj_z` layers 48-63 (+0.5054 GiB), `attn.qg` layers 51/55/59/63 (+0.3369), `mlp.gate_up`
layers 0-1 (+0.4773); a whole layer 0 (+0.5124) or 63 (+0.4843) only as a diagnostic, since it
overlaps the class candidates. Anchor every keep regex with `^text\.layers\.`: an unanchored
`mlp\.down$` or `attn\.o$` also keeps the MTP head's linears.

Caveats:

- The three bf16 out-projections in a rotated container are correct by code (the fold runs before
  the bf16 emit; `convert_reuse` (e-iv) checks the kept bytes are the folded weights) but have never
  been measured. The first KL of such a candidate doubles as that check, and so does its load line
  `r4dx: N linear(s) ... loaded as bf16`, which must read 32 + the candidate's linear count.
- TP=2: only `attn.k`/`attn.v` have tuned TP=2 bf16 rows (`attn.qg`'s 6144x5120 rank shape matches
  a TP=1 bf16 row). Every other kept class runs FallbackTuning at TP=2. KL is measured at TP=1, but a
  TP=2 decode number for a kept pick needs `tune_gemm.py` bf16 rows at the per-rank shapes first.
- Cost: a ~0.48 GiB spender is ~+2.9% weight bytes, ~-3.8% plain decode at the measured 1.23-1.30x
  amplification. At equal bytes it is funded only where its nats/GiB beats the cheapest savers,
  including the two k/v halves at ~0.003.

## 6. GPU runs this plan needs (each handed to the user)

1. Q1 Hessian capture, device 1, server stopped, ~30-60 min.
2. Q1 rung-4 KL runs: pilot (G3), full (G4); decode benchmark for G4.
3. Q2a: bf16 rotated vs unrotated KL (G5), then the quantized KL; ctest + validate/smoke scripts
   (G6); TP=2 check (pre-approved standing permission).
4. Q2b and Q3: KL + decode per candidate.
5. Corpus v2 Hessian capture (3.3) into `D:\models\r4dx\hessian-v2`, after the generator has written
   `D:\models\r4dx\corpus-v2\samples.jsonl`.

## 7. Results (2026-09-25/26, device 1)

Rung 4 against one fresh bf16 reference (`D:\models\r4dx\kl-q1\ref`, `full_logits_golden.py`),
`--layout w4a16`, every candidate converted by `tools/quant2/run_variants.ps1` from the v6 recipe
(`--no-bf16 --mtp on --vision on`, KV calib, `search` + imatrix, attn.k/v kept bf16) plus its own
flags; Hessians `D:\models\r4dx\hessian-v1` (172k tokens, with the rms taps of 3.2).

| container | extra flags | mean KL | top-1 | cpp | english | python | thai |
|---|---|--:|--:|--:|--:|--:|--:|
| v6 | -- | 0.03853 | 91.03% | 0.0241 | 0.0281 | 0.0274 | 0.0745 |
| q1pilot | `--ldlq "mlp\."` | 0.03486 | 90.74% | | | | |
| q1full | `--ldlq .` | 0.02541 | 92.69% | 0.0138 | 0.0206 | 0.0172 | 0.0500 |
| q2a_ldlq | `--rotate q2a --ldlq .` | 0.02480 | 92.86% | | | | |
| **q2ab_ldlq** | `--rotate q2ab --ldlq .` | **0.02261** | **93.23%** | 0.0104 | 0.0194 | 0.0160 | 0.0446 |
| q2ab_d01 | q2ab_ldlq + `--ldlq-damp 0.1` | 0.02359 | 92.30% | 0.0117 | 0.0198 | 0.0169 | 0.0461 |
| q2ab_nokv | q2ab_ldlq, attn.k/v quantized too | 0.02333 | 93.52% | 0.0110 | 0.0197 | 0.0165 | 0.0461 |

Decode (G4/G7): `tools/quant2/bench_decode.ps1`, the Milestone 11 protocol over all four
`tests/model/mtp_prompts.txt` prompts, containers interleaved, 2 runs, host idle (a concurrent
`r4dx-convert` moved plain decode by -6% in an earlier, discarded run); token-weighted aggregate tok/s;
every run reproduced its text byte for byte.

| container | plain | vs v6 | dflash k=7 | acceptance p0/p1/p2/p3 |
|---|--:|--:|--:|---|
| v6 | 36.09 | -- | 109.56 | 24.9 / 66.4 / 28.6 / 55.5% |
| q1full | 36.11 | +0.1% | 105.60 | 28.1 / 53.7 / 25.9 / 54.7% |
| q2ab_ldlq | 35.99 | -0.3% | 105.00 | 26.8 / 63.3 / 23.8 / 53.5% |
| q2ab_nokv | 36.57 | +1.3% | 108.11 | 21.7 / 59.1 / 24.7 / 61.8% |

The DFlash column follows each container's own greedy text (acceptance of one prompt ranges 54-66%
across containers), so four prompts do not resolve a quality effect on acceptance; plain decode is the
per-token cost the gates mean.

Gates: G3 **failed** narrowly (-9.5% vs the -10% bar; KL fell in every segment) -- Q2 went ahead as a
stated deviation because it was already built and LDLQ is expected to need the rotation. G4 **passed**
(q1full -34%, speed-neutral; validate_dflash 3/3 and server smoke 205/0). G5 **passed** (rotated vs
unrotated bf16, 4 layers: 1.7e-5 q2a, 3.3e-5 q2ab). G7 **passed** (q2ab_ldlq -11% vs q1full, -41% vs
v6; plain decode -0.3%). Damping 0.1 is worse than 0.01. Quantizing attn.k/v costs +3% KL for
-0.23 GiB and +1.3% plain decode. Thai stays the worst segment (2x English) and is ~1% of the
hessian-v1 corpus, which is what corpus v2 targets.

G6 on q2ab_ldlq (`tools/quant2/g6_validate.ps1`, logs `D:\models\r4dx\g6-qwen38-27b-q2ab_ldlq`):
validate_dflash w4a16 3/3 byte-identical; validate_spec_sampling w4a16 24/24 (plain vs `--mtp 3` vs
`--dflash k=7`, sampled); smoke `-Mtp 3` 160/0; smoke `-Tp 2 -TpMode emulate -Dflash` 162/0 (real TP=2
not run: it needs device 0); smoke `-Dflash -ToolRoundTrip -Vision` 203/2. Both FAILs are the
"vision multi-turn" yes/no case's prefix reuse, and they are not a rotation defect: this container's
turn-1 answer is the single token `" yes"` (leading space; q1full/v6 answer without it), the chat
template trims replayed assistant content (`|trim`), the re-rendered history therefore differs from the
committed tokens, and the server's all-or-nothing reuse (the GDN state cannot be rewound) correctly
re-prefills. The same smoke's free-form case (a 37-non-ASCII-char Japanese turn 1) REUSED the prefix
without re-encoding the image on this container. The server-side gap (reuse lost after any
whitespace-led reply) is filed as its own task.
