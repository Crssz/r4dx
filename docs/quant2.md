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
| Q3 | libr4d `r4d_gemm_w4a16_nt_m64` group becomes a template parameter (32/64/128 instantiated); group recorded per tensor; allocation by the Milestone 11 nats-per-GiB rule | makes depth/class-aware precision possible without a second format |
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
  manifest sha256 and the list of LDLQ'd bases.

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
`H` exactly rather than re-captured.

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

- libr4d: `R4D_GEMM_W4_GROUP` becomes a template parameter; entry points
  `r4d_gemm_w4a16_nt_m64_g{32,64,128}` plus the existing one (build default) for compatibility.
- Container: `__metadata__.quant.w4a16.groups` maps base name -> group when not all equal; the
  loader dispatches per `QuantLinear`; the tuning table gains a group column; the TP slicer
  already takes the group per tensor (`ShardLoader`'s `w4a16_group_` becomes per call).
- Allocation: the Milestone 11 method, automated -- measure per (class, depth half) the KL
  recovered by g32 / g64 / g128 (one class at a time, rung 4), then fill the byte budget greedily by
  nats per GiB and stop at the cliff.

## 6. GPU runs this plan needs (each handed to the user)

1. Q1 Hessian capture, device 1, server stopped, ~30-60 min.
2. Q1 rung-4 KL runs: pilot (G3), full (G4); decode benchmark for G4.
3. Q2a: bf16 rotated vs unrotated KL (G5), then the quantized KL; ctest + validate/smoke scripts
   (G6); TP=2 check (pre-approved standing permission).
4. Q2b and Q3: KL + decode per candidate.
