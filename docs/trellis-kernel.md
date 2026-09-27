# Trellis weights in production: container, converter, kernels, runtime (design)

Branch `quant2` (worktree `%USERPROFILE%\dev\r4dx-quant2`, libr4d at `99a5d94` on `quant2-groups`).
Written 2026-09-26. This document is design only: nothing here has been run on a GPU. It turns the
EXL3/QTIP trellis format specified in `docs/trellis.md` into something that can be shipped. The oracle
(`tools/reference/trellis_quant.py`, currently running on GPU 1) decides **whether** it ships; this
document decides **how**.

**Notation.** `K` means in_features and `N` means out_features, using r4dx's `W[N, K]`. `KB` is the
number of trellis bits per weight, which `docs/trellis.md` calls K. A tile is 16 k × 16 n.
"Oracle bits" are the `.trellis` / `.suh` / `.svh` tensors written by `quantize-model`
(`trellis_quant.py:1561-1577`), which `reconstruct` (`:1045-1050`) turns into
`W_hat = diag(suh) P_k decode(words) P_n diag(svh)`.

Four checks were run on the CPU while writing this document. They are described in 4.2 and 4.6,
and M0 commits them to the repo:

- A numpy check of the lane map proposed below, compared with `unpack_states` and
  `tensor_core_perm`.
- A device-only compile of the proposed K-loop to gfx1201 assembly. It was compiled only, never
  launched.
- A device-only compile of `clock64()` / `wall_clock64()` (for M1's in-kernel clock), never
  launched.
- A per-SIMD cycle-budget model of decode speed (4.6), in Python.

**Revision 2 (same day)** follows two adversarial reviews (performance and integration). Every
finding was checked against the code; section 9 lists where each one was fixed, and the findings
whose own evidence had to be corrected.

---

## 0. Decisions at a glance

| Question | Decision |
|---|---|
| Bits in the container | The oracle's own bits, imported. The tile grid is re-ordered to n-major pairs. The bits inside each tile, EXL3's position order and the ring encoding are all unchanged. No re-encoding. |
| Converter | **Import, not port.** Add `r4dx-convert --trellis-from <oracle dir>`, pinned by sha256 and checked by a full CPU reconstruction (section 3). |
| Ship point | **KB = 4 uniform (`K4m`, matched Hessian basis).** The 4/5 mix is a fallback only (section 4.6: KB = 5 costs speed and saves under 1% of today's bytes). KB = 3.5 is not supported. |
| Kernel placement | GEMM, decode and FWHT epilogue go in **libr4d** as a new unit, `r4d_gemm_trellis_nt_m64.hip`. The input-transform kernels go in **src/kernels** and the attention kernels. |
| Decode vs verify vs prefill | **One kernel for M = 1..64** using WMMA f16, as w4a16 does. M ≤ 16 always takes the M = 1 tuning, so every verify row is bit-identical to the decode row. There is no dot2 GEMV and no int8-activation GEMV. |
| Lane map | **Column split.** A wave decodes two adjacent n-tiles into fragment F0 (columns 0-7 of both tiles) and fragment F1 (columns 8-15). Each lane needs 3 dwords per tile and 6 `v_alignbit`. The decode costs 3.875 VALU per weight, exact against `codebook_np`, plus 0.25-0.69 of address overhead (4.2). |
| Output Hadamard + svh | **In the GEMM epilogue**, in fp32, with one bf16 rounding. Consumers do not change. When a 128-column group is split across workgroups (WGs), by cross-WG split-K or by WGs narrower than 128 columns, the last WG to arrive sums the partials in a fixed order and applies the transform. |
| Input Hadamard + suh | A separate kernel, one wave per 128-block, that replaces today's per-linear f16 cast. v1 is self-contained inside `ApplyLinear`: `EpilogueForLayout(kTrellis)` stays `r4dx_epilogue_none`, so no layer code changes. M5 fuses the transform into producers, byte-identical to v1. |
| gate/up with different suh | One `mlp.gate_up` linear with **2 parts**. The GEMM reads a different A per part, and each 128-column group lies inside one part. |
| TP = 2 | **Supported.** Every rank slice is 128-aligned. The output transform runs per rank, before the all-reduce, which is valid because the transform is linear. |
| lm_head, MTP head, DFlash drafter, vision | Stay w4a16 (the q2ab_hv2_q3 recipe) or bf16. `Container::Load` maps a requested trellis head layout to w4a16, so every caller gets the same answer. |
| q2ab rotation | Mutually exclusive with trellis. The converter and the loader both refuse the combination. |
| **Speed outlook** | **Marginal, and the expected case fails A3.** With no WMMA/VALU overlap at a sagged clock the model gives a 0-4% plain-decode loss; trellis wins (up to +11%) only if most of its ALU time hides behind memory. M1 is a real kill gate and measures exactly that (4.6). |
| Prefill | Expected 0.74-0.84× of q2ab. M8 (tiled prefill) is planned work, not a contingency. |

---

## 1. Goal and acceptance

**Goal.** Ship a container whose 400 decoder linears are trellis (EXL3 mul1) and everything else
matches today's production container `q2ab_hv2_q3`. It must lower KL at equal or fewer decode bytes
and must not slow decode.

**Gates.** Each is measured on the final container. Numbers in square brackets are today's values
(`docs/quant2.md:1176-1181`, `:1201-1207`; `docs/tp.md:112`).

| # | Gate | Pass condition |
|---|---|---|
| A0 | Oracle pre-gate (`docs/trellis.md:6-7`) | Matched-basis oracle weights-only KL ≤ ~0.008 at `K4m` or `mix4.5m`. If this fails, **nothing below M0 is started.** |
| A1 | Quality | Rung-4 canonical KL (`tool_teacher_forced_logprobs --layout trellis` + `kl_report.py` against `D:\models\r4dx\kl-canon\ref` on `tokens_canon.json`) **≤ 0.0117** (0.75 × [0.01555]), at decode bytes ≤ [13.68 GiB]. Try `K4m` (about 12.13 GiB) first, then `mix4.5m` (about 13.55 GiB). A result between 0.0117 and 0.01555 is a real but small gain that does not justify a second weight format; report it and do not ship. |
| A2 | Implementation fidelity | On a `--lm-head bf16` twin of the container (same body bytes): runtime KL − oracle KL of the same point − 0.0012 (fp8 KV, `docs/quant2.md:1194`) must lie within **±8e-4**, which is 2× the reference self-noise of 3.9e-4 (`docs/quant2.md:64`). Top-1 must be within 0.3 points of the oracle's. Measured on v1 (M4); M5's fused producers are byte-identical to v1 by construction and by test (5.4), so A2 carries over. |
| A3 | Decode speed | `tools/quant2/bench_decode.ps1`, interleaved with q2ab_hv2_q3 in the same run, both on the new binary (A6 shows its w4a16 path is unchanged). Median tok/s must be ≥ q2ab's median − 0.5% for plain [35.9], `--dflash k=7` [108.3] and `--mtp 3` [65.9]. Each mode is predicted from its own round composition (4.6): the drafter is unaffected, and the per-row transform and epilogue scale with M. |
| A3p | Prefill | Tokens/s on the ≥ 256-token prompts of the same run, compared with q2ab: **≥ 0.80×**. The expected value is 0.74-0.84× (4.7), so M8 is planned. |
| A4 | G6 | `tools/quant2/g6_validate.ps1 -Layout trellis` 5/5 (`g6_validate.ps1:25-36`): `validate_dflash`, `validate_spec_sampling` (plain vs `--mtp 3` vs `--dflash k=7`, sampled, bit-exact), smoke `-Dflash -ToolRoundTrip -Vision`, smoke `-Mtp 3`, and smoke `-Tp 2 -TpMode emulate -Dflash`. `g6_validate.ps1` gains a `-Layout` parameter (default `w4a16`) that it passes as `-Layouts` to both validators and as `-Layout` to all three smoke steps; today it hardcodes `w4a16` (`:27`, `:29`) and the smoke steps fall back to `smoke.ps1`'s default (`:138`). The validators and `smoke.ps1` need no change. |
| A5 | TP = 2 | **Supported, not refused.** The emulate smoke in A4 plus `test_tp_loader`'s trellis cases (section 6) must pass. Real TP = 2 needs device 0 and is not run, the same as for q2ab (`docs/quant2.md:1219-1220`). |
| A6 | No w4a16 regression | The change touches shared code (`PickTuning`, `LinearTuning`, `ApplyLinear`, `Container::Load`, `ShardLoader::Linear`, the epilogue host entries). On the final binary with q2ab_hv2_q3: full ctest green; `g6_validate.ps1` (default `-Layout w4a16`) 5/5; `tools/tp/tp1_identity.ps1` SHA-equal to its frozen baseline (`docs/tp.md` 10.3); and generated text SHA-equal to the pre-trellis quant2 binary on `bench_decode.ps1`'s prompts (plain, `--mtp 3`, `--dflash k=7`). |

If A0 or A1 fails the format is not shipped, and M1+ work stops at the first milestone gate that
fails.

**Decision D1: settled by the user's stated priority.** The question: what happens if A1 passes by
a wide margin (say KL ≤ 0.010) but A3 misses by 1-4%? The user's rule (2026-09-26) is "best accuracy
first, then speed, and it must use the native RDNA4 accelerator". So a trellis container that passes
A0-A2 and A4-A6 **ships and becomes the recommended container**, even if A3 misses by at most 4%.
The miss is documented next to the bench numbers, and the w4a16 path stays fully supported.

A decode loss above 4% is not covered by this rule. It goes back to the user with the numbers, and
M5's tuning comes first. The kernel must use WMMA (4.3), so a VALU-only GEMV is not an option even if
it measures faster.

---

## 2. Format in the container

### 2.1 Tensors

Trellis tensors are written only for trellis linears, so every other container stays
byte-identical. Every tensor is U8 with a trailing byte axis, following the convention in
`docs/container-format.md:6-27`.

| Tensor | Shape (U8) | Content |
|---|---|---|
| `<base>.trellis.w` | `[N*K*KB/8]`, 1-D like `.w4a16.wq` | uint32 little-endian ring words in the **pair grid** below |
| `<base>.trellis.suh` | `[P*K, 2]` | fp16 suh, signs folded in. `P` = 2 for `mlp.gate_up` (gate's K values, then up's); 1 for every other linear. |
| `<base>.trellis.svh` | `[N, 2]` | fp16 svh, signs folded in, in container row order (gate then up for `gate_up`) |

**Pair grid.** Word `w` (0 ≤ w < 8·KB) of tile `(tn, tk)` is stored at uint32 index

```
idx(tn, tk, w) = (((tn >> 1) * (K/16) + tk) * 2 + (tn & 1)) * 8*KB + w
```

Tile `(tn, tk)` holds HF weight rows `n = 16tn..16tn+15` and columns `k = 16tk..16tk+15`, which is
EXL3's `W^T` tile `(tk, tn)`. Inside a tile the words are the oracle's exactly: 8·KB uint32 words per
tile, stream bit `S[32w]` at bit 31, and positions in EXL3 tensor-core order (`docs/trellis.md:123-142`,
`:220-243`; `trellis_quant.py:243-255`, `:669-709`).

The oracle stores `[k/16][n/16][8KB]` (`trellis_quant.py:1561`). The pair grid is only a permutation
of whole tiles, so it is lossless. It makes three things contiguous:

- one k-tile of one wave's tile pair: 64·KB bytes (256 B at KB = 4);
- a wave's K slice;
- any 128-aligned run of N rows, which is how tensor parallelism cuts a column-parallel linear
  (2.4).

### 2.2 Which linears are trellis

All 400 decoder linears, as body layout `trellis`:

- `gdn.in_proj_qkv`, `gdn.in_proj_z`, `gdn.out_proj` (48 layers each);
- `attn.qg`, `attn.k`, `attn.v`, `attn.o` (16 layers each);
- `mlp.gate_up`, `mlp.down` (64 layers each).

These stay in their current form:

- **w4a16:** `lm_head`, `mtp.attn.qg/o`, `mtp.mlp.gate_up/down` and `mtp.draft_head.lm_head`, with
  the q3 recipe's groups.
- **bf16:** `mtp.attn.k/v`, `mtp.fc`, `gdn.in_proj_a/b`, embeddings, norms and vision.
- A base selected by `--keep-bf16` becomes bf16-only.

### 2.3 `__metadata__.quant.trellis`

```json
"trellis": {
  "format": "r4dx-trellis", "version": 1,
  "codebook": "mul1",
  "codebook_consts": {"mult": "0x83dcd12d", "k_inv_f16": "0x1eee", "k_bias_f16": "0xc931"},
  "state_bits": 16, "tail_biting": true,
  "position_order": "exl3-tensor-core",
  "bitstream": "ring-u32-msb-first",
  "tile_grid": "n32-pairs-k-major",
  "hadamard": {"block": 128, "order": "sylvester-natural", "scale": "1/sqrt(128)",
               "input": "x*suh then H", "output": "H then *svh"},
  "prescale_log2": 0,
  "linears": {
    "text.layers.0.gdn.in_proj_qkv": {"bits": 4},
    "text.layers.0.mlp.gate_up":     {"bits": 4, "parts": [17408, 17408]},
    "...": {}
  }
}
```

- **Per-linear rate.** `bits` is 4 or 5 per linear. `gate_up`'s two parts always share it, because
  the allocator promotes gate and up together (`trellis_quant.py:116-127`, `:1294-1325`). The loader
  also derives the rate from the byte size and cross-checks it.
- **Hadamard signs.** They are inside suh and svh (`docs/trellis.md` 8.1). No seeds and no sign
  vectors exist at runtime.
- **Prescale.** `prescale_log2` is the power-of-two shift `s` of 4.8, container-wide, default 0,
  written by `r4dx-convert --trellis-prescale-log2` (M4's A-range study picks it; a non-zero value
  means re-running the import, which takes minutes). A linear may override it with its own
  `"prescale_log2"` entry. The loader copies it into `QuantLinear::trellis_prescale_log2`; it is not
  a tuning knob.
- **Alignment.** The file needs none: `ContainerWriter` packs tensors back to back, and the loader
  copies into `hipMalloc` buffers (256-B aligned). The kernel needs 16-B alignment, which every
  device buffer and every arena allocation already has (`arena.hpp`; `linear.cpp:194-218`).

### 2.4 Tensor-parallel slicing

The slicing rules mirror the w4a16 ones (`tp_shard.cpp:55-101`, `:259-284`):

| Tensor | Column-parallel (`kRows`: gate_up, qkv, z, qg, k, v) | Row-parallel (`kCols`: down, o, out_proj) |
|---|---|---|
| `.trellis.w` | one contiguous run per row segment: pair rows `[n0/32, (n0+cnt)/32)` | per pair row, bytes `[(pr*K/16 + k0/16)*64*KB, +kc/16*64*KB)`, i.e. N/32 runs |
| `.trellis.suh` | replicated (all P parts, full K) | split by the rank's K range |
| `.trellis.svh` | split by `RankRows` (per segment) | replicated (full N) |

Every rank range at TP = 2 is 128-aligned:

- gate/up 8704 = 68·128;
- qkv 1024/1024/3072;
- z 3072, qg 6144, k/v 512;
- rank K 8704 / 3072 / 3072.

The 128-alignment matters because both Hadamards and the scale vectors work in 128-blocks. Tile
pairs only need 32-alignment.

### 2.5 Version gating

- **A new binary** parses `quant.trellis` before any upload, next to `ParseRotationMetadata`, in
  **both** loaders: `Container::Load` (`container.cpp:491-494`) and `Container::LoadShard`
  (`:1031-1032`), which parses rotation separately. The parsed spec is passed into `ShardLoader`,
  whose `kTrellis` case needs each linear's `bits` and `parts`. Both loaders refuse:
  - any unknown `format`, `version`, `codebook`, `position_order`, `bitstream`, `tile_grid` or
    `hadamard` value;
  - a `bits` value the kernel does not instantiate (`r4d_gemm_trellis_nt_m64_has_rate`);
  - a `.trellis.w` without a `linears` entry, and a `linears` entry without its three tensors;
  - byte sizes other than `N*K*bits/8`, `P*K*2` and `N*2`;
  - any `K`, part or `N` not divisible by 128 (and, in `LoadShard`, any rank range of a trellis
    linear not divisible by 128; this check belongs in the loader or in `tp::PlanRows`/`PlanCols`,
    not in `ModelConfig::Shard`, which knows neither the container nor the layout);
  - `rotation` and `trellis` together;
  - loading with `--layout` other than `trellis` (the message says "this container's body is trellis:
    run with --layout trellis");
  - `--layout trellis` on a container without the block.
- **An old binary** finds neither `.w4a16.*` nor `.bf16.w` for a body linear and throws
  "no tensor found" (`container.cpp:336-337`). This works only because the converter never writes
  `.bf16.w` for a trellis linear (3.4), and it is the whole compatibility guard. `r4dx_format_version`
  is not read by anything and is not bumped.

**Bytes.** "Decode bytes" follows `docs/quant2.md:1172-1174`: every text-layer weight plus
`lm_head`, measured from the file. For `K4m` that is 11.341 GiB of decoder linears
(`docs/trellis.md:571`, suh/svh included), a w4a16 g32 lm_head of 0.740 GiB, and about 0.049 GiB of
other text-layer weights that stay bf16 (`gdn.in_proj_a/b` about 0.044, conv1d about 0.004, norms,
`A_log`, `dt_bias`): **about 12.13 GiB**. `mix4.5m` is **about 13.55 GiB**. q2ab_hv2_q3 is 13.68 GiB.
M3 measures the container with the same method used for the 13.68 figure.

---

## 3. Converter

### 3.1 Import, not port

1. **The shipped bits are the measured bits.** Gates A1 and A2 compare against the oracle's KL, and
   only an import guarantees that the tile values are the ones whose KL was measured.
2. **A port cannot reproduce the oracle.**
   - The LDLQ feedback's fp32 GEMM order differs between CPU and torch-on-GPU, so the states diverge.
   - A C++ port would therefore need its own KL run.
   - It would be days of work: block LDL, backward 16-row LDLQ, the g-scale search, the refit, and
     the Viterbi.
   - Only `HessianStore` and `dense_linalg` carry over.
3. **Cost.** The converter is GPU-free by decision (`quant2.md:27`). A CPU port would take 3-5 h
   per conversion; the encoder alone is 2.3-3.9 h per rate (`docs/trellis.md:824-827`). An import
   takes minutes.
4. **Reproducibility.** The container is pinned to specific files by sha256, and the oracle's
   encoders are bit-identical across HIP, CPU and torch (`docs/trellis.md:773-776`).
   Re-quantizing for new Hessians or a new checkpoint means re-running the oracle, 1.3-1.5 h per rate
   on one R9700.

### 3.2 CLI

```
r4dx-convert --input <HF dir> --output D:\models\r4dx\qwen38-27b-trellis-k4m.r4dx `
  --trellis-from D:\models\r4dx\trellis-q\K4m            # override dir or its weights_override.json
  [--trellis-manifest-sha256 <hex>]                      # pin: refuse if the manifest hashes otherwise
  [--trellis-verify full|none]                           # default full (3.3 step 10)
  [--trellis-allow-basis exl3]                           # default: refuse hessian_basis != "matched"
  [--trellis-prescale-log2 <int>]                        # default 0 (2.3, 4.8)
  --rotate none --no-bf16 --mtp on --vision on --kv-calib D:\models\r4dx\qwen38-27b.kvcalib-full.json `
  --layouts w4a16 --lm-head w4a16 <q2ab_hv2_q3's lm_head/MTP flags: --quant search, --ldlq, --hessian-dir,
  --w4a16-group-rule "^lm_head$=32", ...>
```

`--layouts` and `--lm-head` keep their current meaning. Now they govern only the linears the manifest
does not cover: `lm_head`, `mtp.*` and the draft head.

### 3.3 Checks, all before anything is written unless noted

**Plan phase** (before `FinalizeHeader`):

1. The manifest (`trellis_quant.py:1360-1369`, `:1598-1619`; a mix's at `:1667-1677`) must have:
   - `format` = `r4dx-weights-override`, `version` 1, `encoding` = `trellis-exl3`;
   - `complete` true and `missing_count` 0;
   - no layer used by this conversion listed in `stale_layers`. A mix manifest has no
     `stale_layers`, `layers_done` or top-level `code_sha256`; a missing `stale_layers` means empty,
     and the code sha is read from `allocation.source_code_sha256` instead.
   - Per-tensor `file` paths may be absolute (a mix's always are, `:1660`) or relative to the
     manifest's directory (`tensor_file`, `:1372-1374`).
2. `codebook` = mul1 and `hessian_basis` = `matched`. `exl3` is accepted only with
   `--trellis-allow-basis exl3`, because `docs/trellis.md` 14 (b) measured the matched basis 7-73%
   better on q/k/v.
3. `config_sha256` must equal the sha256 of the checkpoint's raw `config.json` bytes: the same hash
   the converter writes as `__metadata__.config_sha256`, and the one the oracle writes
   (`trellis_quant.py:1521`).
4. Per HF tensor:
   - `encoding` = `trellis-exl3`;
   - `K` ∈ {4, 5}, compared numerically (a mix stores it as a float, 4.0 or 5.0; 3.5 is refused
     because the kernel does not instantiate it);
   - `k`, `n` and `shape_hf` equal the checkpoint's `[n, k]`;
   - `words_shape` = `[k/16, n/16, 8K]`;
   - gate K = up K in every layer.
5. **Coverage.**
   - Every body linear's HF names must be in the manifest (`add_linear`'s `hf_names`, which are
     exactly the oracle's keys, `trellis_quant.py:143-144`).
   - A base matched by `--keep-bf16` is logged and skipped.
   - Partial coverage is refused: there is no mixed w4a16/trellis body in v1. A mixed body is a
     possible later speed lever (4.6), behind an explicit flag and its own KL run.
6. `--rotate` must be `none`. `--trellis-from` with `--selftest`, `--dflash-gguf`,
   `--reuse-tensors-from` or `--record-reuse-guard` is an argument error.

**Emit phase:**

7. The sha256 of each distinct `L<ii>.safetensors` must equal `rec.file_sha256`. Each file is hashed
   once and cached (`trellis_quant.py:1575-1577`). A mix manifest points at absolute files through
   `tensor_file` (`:1372-1374`).
8. Tensors are read through `SafetensorsReader`, with dtype and shape checked: I32
   `[k/16, n/16, 8K]`, F16 `[k]`, F16 `[n]`.
9. **Regrid.** Words are copied into the pair grid with the tile permutation only. The bits are
   untouched; parts are concatenated along N; suh is written as `[P][K]`.
10. **`--trellis-verify full`** (default). This step decodes the **container bytes just written**,
    not the source files, so the regrid is covered too.
    - It reconstructs `W_hat` on the CPU, 32 threads, fp32 with fp64 reductions, and computes
      `rel = ||W_hat - W||/||W||` against the bf16 checkpoint.
    - It refuses unless `|rel - rec.rel_weight_err| <= 0.02*rec + 1e-4`, the tolerance of
      `full_logits_golden.py`'s override check.
    - Cost is about 4e11 flops plus reading about 50 GB: a few minutes.
    - `none` exists for debug builds only.

**Metadata written:**

- `quant.trellis` (2.3);
- a `quant_summary` line such as "body: trellis mul1 KB=4 x400 (imported)";
- `r4dx_convert_run.trellis` = {manifest path and sha256, `encoding`, `bpw_target`, `K_uniform`,
  `hessian_basis`, `hessian_manifest_sha256`, `config_sha256`, `code_sha256` (or a mix's
  `allocation.source_code_sha256`), `recipe`, `files: {name: sha256}`,
  `verify: {mode, worst |rel-rec|/rec}`}.

Code sha mismatches between the oracle's sources are **recorded, not refused**: the bits describe
themselves and step 10 checks them.

### 3.4 Interplay and code changes

- **`--rotate q2ab` is refused.** A trellis linear built from the unrotated W would be fed `x·Q`
  with the norms removed, which is garbage. The runtime cannot rotate per linear (`container.h:241`;
  `model.cpp:128-145`). Trellis's own RHT takes over q2ab's incoherence role.
- **`--keep-bf16`.** The kept linear gets `KeptBf16LayoutSet()` (`linear_layouts.hpp:58`) and its
  manifest entry is skipped. At load, the tier-2 fallback serves it (`container.cpp:325-327`).
- **`--no-bf16` is implied for trellis linears.** Their `LayoutSet` is `{trellis}` only, whatever
  `--layouts` says, so no `.bf16.w` can defeat the old-binary guard (2.5).
- **Linears not in the manifest** (lm_head, mtp, draft head) go through the existing w4a16 path, so
  `--ldlq`, `--imatrix` and `--w4a16-group-rule` apply to them. `HasQuantizedLayout` (`main.cpp:598`)
  stays 4-bit-only, so LDLQ never touches a trellis linear. With the same flags, these linears'
  bytes are identical to q2ab_hv2_q3's, because they were never rotated.
- **Reuse guard.** Refused in v1: an import is I/O-bound and has nothing to reuse. To support it
  later:
  - bump `kReuseGuardVersion` (`reuse_guard.hpp:59`);
  - add the manifest sha and file shas to `BuildReuseGuard` (`main.cpp:1103`);
  - erase `quant.trellis.linears` before the baseline's quant equality check (`main.cpp:1237`).

**Code changes:**

| File | Change |
|---|---|
| `linear_layouts.hpp` | `LayoutSet` (`:31`) gains `trellis`, `trellis_bits` and parts. Also extend `LinearLayoutTensorNames` (`:69`), `LayoutSetId`/`ParseLayoutSetId` (`:96-139`, token `trellis.k4`/`trellis.k5`), `LinearLayoutBytes` (`:146`) and `PlanLinearLayouts` (`:182`; checks K and every part ÷128). |
| New `trellis_import.hpp` | `TrellisSource` next to `LdlqSource`/`RotationSource` (`main.cpp:621`, `:846`): manifest parse, checks 1-8, regrid emit, verify. |
| `main.cpp` | `AppArgs`/`ParseArgs` (`:314-536`) get the five flags. `add_linear` (`:1715-1783`) resolves in the order w4a16 group → trellis (if covered) → `--keep-bf16`; a covered linear's plan/emit jobs call `TrellisSource`. `BuildQuantMetadata` (`:263`) and `r4dx_convert_run` get the trellis blocks. |

`compare_containers.py` works on trellis containers unchanged.

---

## 4. Kernels

### 4.1 Where they live

| Piece | Home | Why |
|---|---|---|
| GEMM (decode, verify and prefill), the tile decode helper, the FWHT-128 epilogue, and the test entries (`_raw`, `r4d_trellis_reconstruct_f16`) | **libr4d**: `r4d_gemm_trellis_nt_m64.hip`, `r4d_trellis_dq.h`, `r4d_fwht128.h` | Every GEMM lives there. `tune_gemm.py` times through `r4d.pyd`, and the WMMA fragment helpers are in `r4d_gdn_wmma.h:2-6`, `:79-83`. |
| Input transforms (`x⊙suh`, FWHT-128, f16) | **src/kernels** (`trellis_transform.hip`) and `src/model/attention` | They fuse into producers r4dx already owns, and those producers carry the q2ab Hadamards (`r4dx_kernels.hip:477-538`, `attn_kernels.hip:52-127`, `rotate_residual.hip:96-155`). |

**libr4d checklist** (`README.md:117-129`):

1. the new unit;
2. declarations in `r4d.h` (next to `:253-303`, `:407-440`);
3. a registry row with op `gemm_nt` and dtype `trellis_mul1_k{4,5} x f16 -> bf16` (`r4d_registry.hip:110-128`);
4. pybind in `r4d_module.hip:659-690`;
5. `build.sh:38-60` and `build_windows.ps1:164-186` UNITS;
6. `test_trellis_gemm.py`, following `test_w4a16_gemm_groups.py`.

**r4dx side:**

- `third_party/CMakeLists.txt:54-70` `R4D_UNITS`, and a `GemmTrellisNtM64` wrapper in `r4d.hpp`
  next to `:188-202`.
- `src/kernels/CMakeLists.txt` lists every hipcc translation unit explicitly (`:46-137`):
  `trellis_transform.hip` needs its own `add_custom_command` (the same flags as
  `r4dx_kernels.hip`, WGP mode) and an entry in `add_library(r4dx_kernels ...)` (`:137`).

**Submodule process (M0).**

- The quant2 pin `99a5d94` exists only in the worktree's module clone. It is not in the canonical
  `C:\Users\pay20\dev\libr4d`, nor in main's clone.
- Branch `trellis` off `99a5d94`, then push both to the canonical clone before quant2 merges.
  Otherwise a fresh checkout cannot resolve either pin.
- `.gitmodules` still names `C:/Users/user/dev/libr4d`, a path on another machine; that is a separate
  fix and not part of this work.

### 4.2 The lane map: EXL3 order on RDNA4 WMMA, with no re-encode

**The mismatch.**

- EXL3 positions `8t..8t+7` are one NVIDIA lane's **2 columns × 4 k**. For column c < 8 they are
  `32c + 8m + j`, j = 0..3, m = 0..3, with `k = 2m + {0,1,8,9}[j]`. Column c + 8 is the same set
  shifted by 4.
- A gfx12 WMMA B fragment gives lane L **1 column × 8 k**: `n = L%16`,
  `k = 8(e>>2) + 4(L>>4) + (e&3)` (`r4d_gdn_wmma.h:2-6`).
- So 8 consecutive positions always span two columns, and a straight decode would need cross-lane
  shuffles.

**The column split.**

- A wave owns a **pair** of adjacent n-tiles (t0, t1) and builds two fragments per k-tile:
  - F0 = columns 0-7 of t0 (lanes 0-7) and columns 0-7 of t1 (lanes 8-15);
  - F1 = columns 8-15 of both tiles.
- Output columns are permuted inside the pair, and the epilogue writes them back to their true
  positions (4.5).

**What lane L needs.** Let `t = (L>>3)&1`, `c = L&7`, `h = L>>4`.

- Runs `m = 2h` and `m = 2h+1` of column c lie in ring words `wa = 4c+2h` and `wa+1` of tile t.
  - Positions 0-3 of each word belong to F0.
  - Positions 4-7 of each word belong to F1: they are column c + 8's, which also sit in words `wa`
    and `wa+1`.
- The previous word `(wa-1) mod 32` supplies the 12 history bits. The mod-32 wrap is tail-biting.
- For K = 4, each state is 16 bits ending at position p's own nibble, so in the 96-bit window
  `P:A:B` the lane's 16 states sit at every 4-bit offset. A pair of states 16 bits apart is exactly
  one `v_alignbit`:

```
RA[j] = alignbit(P, A, 12-4j)   j = 0..2,   RA[3] = A     hi16 = F0 state (A, j), lo16 = F1 state (A, j)
RB[j] = alignbit(A, B, 12-4j)   j = 0..2,   RB[3] = B     hi16 = F0 state (B, j), lo16 = F1 state (B, j)
fragment element order (F0 and F1 alike): e0,e1 = A j0,j1; e2,e3 = B j0,j1; e4,e5 = A j2,j3; e6,e7 = B j2,j3
```

So the lane loads one b64 (`A`, `B`; `wa` is even) and one b32 (`P`), and extracts 16 states with 6
ALU ops (0.375 per weight). No cross-lane traffic is needed, and the four packed f16 dwords come out
in fragment order, so neither A nor B needs a k-permutation.

**KB = 5.** Tiles are 40 words.

- Lane (t, c, h) loads 5 dwords `ring[(5c-2+3h+i) mod 40]` and pre-aligns them with
  `V_i = alignbit(W_i, W_{i+1}, 16h)`.
- State q (position `32c+16h+q`) is then big-endian bits `[21+5q, 37+5q)` of `V0..V3` for every lane.
- F0 takes q = {0,1,8,9,2,3,10,11} and F1 takes q = {4,5,12,13,6,7,14,15}.

**CPU verification** (session scratchpad; committed in M0 as `tools/reference/trellis_lane_check.py`):

- For 64 random tiles, all 32 lanes × 8 elements × 2 fragments decode **0 mismatches** against
  `unpack_states` + `tensor_core_perm_inv`, at both KB = 4 and KB = 5.
- The value path is **bit-exact against `codebook_np` for all 65,536 states**:
  - the 16-bit split hash;
  - `v_sad_u8`/`v_sad_hi_u8` packing;
  - `v_pk_fma_f16`.
- The affine-deferred variant (4.6) deviates from the codebook by at most 9.77e-4 per weight.

**ISA probe** (device-only `hipcc --cuda-device-only -S --offload-arch=gfx1201`, never launched):

- The assembler accepts `v_mad_u32_u16 ... op_sel:[1,0,0,0]`, `v_pk_mad_u16 ... op_sel_hi:[1,0,1]`,
  `v_sad_u8` and `v_sad_hi_u8`.
- The **decode subset** of the K = 4 step compiles to exactly **6 `v_alignbit` + 16
  `v_mad_u32_u16` + 16 `v_pk_mad_u16` + 8 `v_sad_u8` + 8 `v_sad_hi_u8` + 8 `v_pk_fma_f16` + 2
  `v_wmma_f32_16x16x16_f16`** per lane per (tile pair, k-tile). That is 62 VALU per 16 weights, or
  **3.875 per weight**.
- The **whole probe loop** (`.LBB0_2`) has **73 VALU**: the 62 above plus 11 of address arithmetic
  (4 `v_add_co_u32`, 4 `v_add_co_ci_u32`, `v_ashrrev_i32`, `v_lshlrev_b64`, `v_add_nc_u32`). That
  is 0.69 op/w of overhead, not the 0.25 the first revision assumed. The M1 kernel must keep the
  weight base in SGPRs with a loop-invariant VGPR lane offset and immediate offsets (a scalar pointer
  bump per k-tile, as the w4a16 kernel's constant-stride loop does), and M1 reports the whole-loop
  VALU count of every instantiation, not the decode subset.
- 44 VGPRs, no scratch.

**Consequence.** EXL3's order costs nothing on RDNA4, so the oracle's bits are kept for good. A
WMMA-native re-encode would save no ALU.

### 4.3 GEMM: `r4d_gemm_trellis_nt_m64` (one kernel, M = 1..64)

```c
// r4d.h. Device pointers are int64 (libr4d convention); throws std::runtime_error on a bad shape.
void r4d_gemm_trellis_nt_m64(int64_t a0, int64_t a1, int n_split,   // A parts, f16 [M][K] row stride K, ALREADY
                                                                    // input-transformed; cols >= n_split read a1
                             int64_t w, int64_t svh, int64_t c,     // pair-grid words; fp32 svh [N]; bf16 C [M][N]
                             int64_t ws, int64_t tickets,           // fp32 [SKG][M][N] (split groups only); u32 [N/128]
                             int M, int K, int N, int KB,           // KB in {4, 5}
                             int WV, int SK, int MT, int NP, int SKG, int U, int NT,
                             float out_scale, int64_t stream);      // out_scale = 2^-s / sqrt(128)
int    r4d_gemm_trellis_nt_m64_has_rate(int KB);
int    r4d_gemm_trellis_nt_m64_max_m(void);                          // 64
size_t r4d_gemm_trellis_nt_m64_ws_bytes(int M, int N, int SKG);     // SKG * M * N * 4; used whenever a
                                                                    // 128-group spans > 1 WG
size_t r4d_gemm_trellis_nt_m64_tickets_bytes(int N);                // (N / 128) * 4 (added in M2)
void   r4d_gemm_trellis_nt_m64_zero_tickets(int64_t tickets, size_t bytes, int64_t stream);
                                                                    // hipMemsetAsync; the 4.5 reset
// Test and diagnostic entries (same device decode code):
void r4d_gemm_trellis_nt_m64_raw(/* same minus svh/out_scale; fp32 C, no output transform */);
void r4d_trellis_reconstruct_f16(int64_t w, int64_t q /* f16 [K][N] */, int K, int N, int KB, int64_t stream);
```

**Geometry.**

- A WG has `WV × SK` waves (≤ 1024 threads). Wave `w` owns column block `w / SK` and K slice
  `w % SK`, the same convention as w4a16 (`r4d_gemm_w4a16_nt_m64.hip:180-187`).
- A wave covers `NP` tile pairs, i.e. `32·NP` columns.
- A WG covers `Wc = WV·NP·32` ∈ {32, 64, 128, 256} columns. A 128-group lies inside one A part
  either way. When `Wc < 128`, a 128-group spans `128/Wc` WGs, which meet through `ws` and the
  ticket exactly like cross-WG split-K (4.5). The narrow widths exist because a 128-wide WG gives
  only 40 WGs on the N = 5120 shapes against 32 WGPs (w4a16 runs them with 80-320 WGs), and a
  40-WG grid quantizes badly.
- `SKG` WGs split K across the grid.
- Grid: `(N / Wc, SKG, ceil(M / (16·MT)))`.
- Each wave's K range is `K / (SK·SKG)`, taken in `U` unrolled k-tiles. `U` ∈ {2, 4} is a template
  parameter chosen at launch from the `U` argument, like w4a16's NPW; it is in the tuning table.
- Rows ≥ M are clamped on load and skipped in the epilogue. Columns are exact because N ÷ 128.

**Legality** (the host throws):

- M in 1..64; K and N ÷ 128; `n_split` ÷ 128 and ÷ `Wc` when `Wc` = 256;
- `(K/16) % (SK·SKG·U) == 0`;
- `WV·SK·32 ≤ 1024`;
- LDS (4.5, per 8-row reduction pass): `SK · Wc · 8 · 4 B ≤ 64 KiB`, i.e. SK ≤ 16 at `Wc` = 128
  and SK ≤ 8 at 256. It does not grow with MT;
- `NP·U ≤ 8` (the register budget in 4.4), and every instantiation's VGPR count ≤ 190, checked at
  build time from the code object's `.vgpr_count`;
- a split 128-group (`SKG > 1` or `Wc < 128`) only when `gridDim.z == 1`, so tickets are only ever
  per 128-group;
- `KB ∈ {4, 5}`.

**Shape headroom.** Rank K 8704 = 544 k-tiles = 2^5·17, which caps `SK·SKG·U` at 32 on down's TP
slice; 5120 and 6144 allow 64. The 256-wide WG needs N ÷ 256, which holds for every N and part of
this model at TP = 1 and TP = 2 (5120, 1024, 10240, 6144, 12288, 34816, 17408, 3072, 512, 8704).

**Row identity.**

- For M ≤ 16 the tuning is the M = 1 band's `WV/SK/MT/NP/SKG/U` (`linear.cpp:137-145` extended).
  One row tile is a single WMMA M-tile, and the in-WG, cross-WG and FWHT summation orders do not
  depend on M or on the other rows. So verify row r is bit-identical to decode.
- Only `NT` (a cache hint) may differ per band, the same rule w4a16 follows (`linear.cpp:19-33`).

### 4.4 Inner loop (per lane; K = 4 exact, then K = 5)

```
// ---- loop-invariant, per lane L ----
t = (L>>3)&1;  c = L&7;  h = L>>4
wa = 4c + 2h;  wp = (wa + 31) & 31                  // tail-biting wrap
off_ab = (t*32 + wa)*4;  off_p = (t*32 + wp)*4       // bytes inside a (pair, k-tile) block of 64*KB bytes
KMLO = 0x0000D12D (SGPR); KMHI = 0x83DC0000 (SGPR)   // 0x83DCD12D = KMHI:KMLO
ONESBIAS = 0x64006400; KINV2 = 0x1EEE1EEE; KBIAS2 = 0xC931C931 (SGPRs)

// ---- K loop: kt over this wave's k-tiles, U-unrolled, loads for kt+U issued before decoding kt ----
for p in 0..NP-1:                                     // tile pairs of this wave
  blk = W + ((pair0 + p) * (K/16) + kt) * 64*KB
  {A,B} = global_load_b64(blk + off_ab)   [NT: nt/slc hint]
  P     = global_load_b32(blk + off_p)
  RA0 = alignbit(P,A,12); RA1 = alignbit(P,A,8); RA2 = alignbit(P,A,4); RA3 = A
  RB0 = alignbit(A,B,12); RB1 = alignbit(A,B,8); RB2 = alignbit(A,B,4); RB3 = B
  for F in {F0: half = hi, F1: half = lo}:            // op_sel is per instruction, uniform per F
    for R in {RA0..RA3, RB0..RB3}:
      tlo  = v_mad_u32_u16(R.half, KMLO, 0)           // s * 0xD12D, 32-bit exact
      x_R  = v_pk_mad_u16(KMHI, R.(half), tlo)        // hi16 += s*0x83DC; lo16 += 0  -> x = s*0x83DCD12D
    d0 = v_sad_hi_u8(x_RA1, 0, v_sad_u8(x_RA0, 0, ONESBIAS))   // (1024+sum_A0 | 1024+sum_A1) as f16 bits
    d1 = v_sad_hi_u8(x_RB1, 0, v_sad_u8(x_RB0, 0, ONESBIAS))
    d2 = v_sad_hi_u8(x_RA3, 0, v_sad_u8(x_RA2, 0, ONESBIAS))
    d3 = v_sad_hi_u8(x_RB3, 0, v_sad_u8(x_RB2, 0, ONESBIAS))
    bF = v_pk_fma_f16({d0,d1,d2,d3}, KINV2, KBIAS2)            // == codebook_np bit for bit (one rounding)
    for mt in 0..MT-1:
      acc[p][F][mt] = wmma_f32_16x16x16_f16(afrag[mt][kt], bF, acc[p][F][mt])
// afrag[mt][kt]: two global_load_b64 per lane from A (L2/MALL-resident, fragA as w4a16 :112-117),
// loaded once per k-tile and reused by all 2*NP WMMAs. A part = (block's first column >= n_split) ? a1 : a0.
```

**K = 5** has the same structure with a different fetch and extract:

```
wb = 5c - 2 + 3h
W0..W4 = 5 x global_load_b32(blk + (t*40 + ((wb+i) mod 40))*4)   // per-lane offsets precomputed
V_i = alignbit(W_i, W_{i+1}, 16h)   i = 0..3                        // 4 ops
for q in 0..15:  r = 21 + 5q; i = r>>5; s = r&31
    state_q = s <= 16 ? (V_i >> (16-s))       // s = 0: hi16 via op_sel, free
                      : alignbit(V_i, V_{i+1}, 48-s)
// F0 elements = q {0,1,8,9,2,3,10,11}; F1 = q {4,5,12,13,6,7,14,15}; hash/sad/fma as above (lo16)
```

This comes to 19 extract + 32 hash + 16 sad + 8 fma = 75 VALU per 16 weights, **4.69 per weight**.

**Registers.** The K = 4 probe with NP = 1, U = 1 uses 44 VGPRs. The budget for NP = 2, U = 4 with a
one-step prefetch:

- 48 load VGPRs (3·NP·U·2; 5·NP·U·2 at KB = 5);
- 32 accumulators (16·NP·MT at MT = 1);
- 16 A-fragment VGPRs, plus 16 more if the A fragment is prefetched a step ahead;
- about 30 temporaries and 8 addresses.

That is about 134 VGPRs (150 with A prefetch), or 11 (10) waves/SIMD. The cap is ≤ 190, which is
8 waves/SIMD (`docs/r9700.md:54`, C14). Not every knob combination fits: NP = 4 with U = 4 needs
96 + 64 + 16 + ~38 ≈ 214. Hence the legality rule `NP·U ≤ 8` (4.3) and the build-time check of every
instantiation's `.vgpr_count`. Prefill: MT = 4, NP = 1 is about 150 VGPRs; MT = 4, NP = 2 at U = 1
is about 195 (trellis has no per-group `g_acc`, so this is close to w4a16's 192 at M = 64). Never
set a min-waves `__launch_bounds__` (`r9700.md:707-708`).

**As built (M1, compile only).** `r4d_gemm_trellis_nt_m64_raw` at KB = 4, gfx1201, from the build's
own listing (`tools/reference/trellis_isa_report.py`):

- **Whole K loop: exactly 62 VALU per (tile pair, k-tile), 3.875 per weight, in every
  instantiation**: the decode subset and nothing else. The weight and activation bases are
  wave-uniform SGPRs bumped once per step and every load is `global_load` saddr + a loop-invariant
  lane offset + an immediate, so the probe's 11 address VALU (0.69 op/w) are gone. Two compiler
  behaviours had to be defeated for that, each with an empty inline asm that costs no instruction:
  loop strength reduction turning the bases into per-lane 64-bit pointers, and LICM hoisting the
  lane offset's zero-extension out of the loop (the saddr pattern needs it in the load's block).
- **Decode schedule (after the M1 review).** The decode of one (pair, k-tile) is a single inline-asm
  block (`r4d_trellis_k4_decode`) whose internal dependencies are all at least 8 VALU apart and
  which ends in `s_delay_alu instid0(VALU_DEP_1)`. The first build used one asm per hash: LLVM
  inserts `s_delay_alu` only between instructions it generated, gfx12 then stalls the whole SIMD's
  VALU on each near dependency, and the scheduler left 4-249 of them per K loop (a different number
  per instantiation); the pipe bench's decode loop issued at 0.50-0.52 VALU per clock at 1, 2 and 4
  waves per SIMD. The build check now also fails on any VALU that reads an asm result 1-3 VALU
  later without an `s_delay_alu` between. Measured at M = 1 (weights streamed from DRAM, 10
  shape/tuning pairs, bit-identical outputs): 1.4-19% faster than the per-hash version, most where
  the old schedule was worst.
- **Prefetch:** MT <= 2 double-buffers one step ahead (two named register stages, no copies at the
  loop edge), with `sched_barrier`s pinning each stage's loads ahead of the other stage's decode:
  without them the scheduler sank the loads toward their use, at (NP, U) = (4, 1) past the other
  decode (the loop opened on `s_wait_loadcnt 0`; 1.46x the time). (NP, U, MT) = (2, 4, 2) and
  (4, 1, 2) take one stage (accumulators plus two stages exceed 128 VGPRs; really double-buffered
  they reached 192 and scratch), as does MT >= 3, where each fragment already feeds 3-4 WMMA per
  pair. At an even step count the last issue re-reads the final step: peeling it off measured
  0-1.2% slower on 9 of 10 shapes, so the redundant, cache-resident load stays.
- **Instantiated:** NP·U <= 8 and, per (NP, U), MT up to 4 at NP = 1 and at (2, 1), 3 at (2, 2) and
  (2, 4), 2 at (4, 1), 1 at (4, 2) (set when the next MT up spilled or reached 191-192 VGPRs; with
  the one-block decode it still spills at (4, 1), and would fit at (2, 2), (2, 4) and (4, 2)). 50
  kernels, 54-188 VGPRs, no scratch (checked at build time by `third_party/check_trellis_isa.cmake`);
  the prefill (MT, NP) = (4, 2) at U = 1 is 182.

**Tuning knobs.** All go into the tuning table (`LinearTuning`, 5.3) except `AFFINE`, which is a
compile-time variant:

| Knob | Range | Default and notes |
|---|---|---|
| `WV` | {1, 2, 4} | |
| `NP` | {1, 2, 4} | `Wc = WV·NP·32` ∈ {32, 64, 128, 256} |
| `SK` | {1, 2, 4, 8, 16} | in-WG split; LDS rule in 4.3 |
| `SKG` | {1, 2, 4, 8} | cross-WG split, only when `gridDim.z == 1` |
| `MT` | {1..4} | |
| `U` | {1, 2, 4} | K unroll; template parameter chosen by the launch argument; `NP·U ≤ 8` |
| `NT` | {0, 1} | |
| `AFFINE` | 0 / 1 | skip `v_pk_fma`; 4.6 |

The prescale `s` is not a knob: it comes from the container (2.3).

**Fallback tuning** (`FallbackTuning` gains M, 5.3):

- M ≤ 16: WV = 4, NP = 1, SK = 2, MT = 1, U = 2, NT = 1, with `SKG = clamp(128 / (N/128), 1, 4)`
  rounded down to a power of two and reduced until `(K/16) % (SK·SKG·U) == 0`. (Revision 2 capped
  SKG at 8, which only attn.k/v reached; 10.1 dropped SKG = 8 from the tunings. The kernel still
  accepts it, and `test_trellis_gemm` keeps it deterministic and row-identical.)
- M > 16: SKG = 1, MT = min(4, ceil(M/16)), WV = 4, NP = 1, SK = 2, U = 2, NT = 0.

Without the M > 16 rule, every prefill chunk of a linear with N < 16384 (k, v, z, o, out_proj,
down) would get SKG > 1 with `gridDim.z > 1` and throw. That is the path M4's KL runs take (64-row
chunks, before the M5 tuning sweep), as do a fresh checkout's empty table and the 4-layer test
containers.

### 4.5 Epilogue: in-WG reduce, cross-WG ticket, FWHT-128, svh, bf16

```
// acc element e of fragment F, pair p, row tile mt, on lane L:
//   row m = 16 mt + 8(L>>4) + e;  col n = 32*(pair0+p) + 16*((L>>3)&1) + 8*F + (L&7)   (true column)
1. for each row tile mt, in passes of 8 rows (rows 0-7 = lanes 0-15; rows 8-15 = lanes 16-31;
   the second pass is skipped when every row of the tile past 8 is >= M, so M <= 8 costs one pass):
     LDS red[SK][8][Wc] fp32 <- each wave's accs for these 8 rows at their TRUE columns; barrier;
     S[m][n] = sum_{s=0..SK-1} red[s][m][n] in that order (as w4a16 :334-358, which also reduces
     one row tile at a time); barrier.
   LDS is SK*Wc*32 B whatever MT is (SK*4 KiB at Wc = 128), so SK = 16 is legal at Wc <= 128.
2. if the 128-group is split (SKG > 1 or Wc < 128); c = SKG * max(1, 128/Wc) contributors:
     ws[blockIdx.y][m][n] = S for m < M and this WG's Wc columns  (fp32, arena scratch)
     __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent"); barrier
     thread 0: old = atomicAdd(&tickets[group], 1)  -> LDS broadcast   // group = n0 / 128
     if old != c-1: return                                 // nobody ever waits: no deadlock
     thread 0: tickets[group] = 0                          // self-resetting, capture-safe
     __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent")
     S[m][g*128..+127] = ws[0][m][..] + ws[1][m][..] + ... + ws[SKG-1][m][..]
                                                           // fixed order, own slot included
3. for each (m < M, 128-group g this WG finishes), one wave per (m, g), round-robin over the WG's
   waves:
     y = Fwht128Wave(S[m][g*128 .. +127])                  // 4.8, fp32, stage order lg = 0..6
     C[m][n] = bf16_rn(y[n] * svh[n] * out_scale)          // out_scale = 2^-s / sqrt(128)
```

The summation order of steps 1-2 depends only on (SK, SKG, Wc), never on M or on which WG arrives
last, so the row identity of 4.3 holds.

- **Where the tickets and workspace come from.** All of a container's tickets live in **one**
  device buffer owned by the `Container` (`uint32` per 128-group per trellis linear, each
  `QuantLinear` holding a pointer into it), zeroed at load. Each TP rank loads its own `Container`,
  so each rank has its own tickets. `Container::ZeroTrellisTickets(stream)` is one
  `hipMemsetAsync`; `Model::Reset()` calls it before its existing synchronize, and
  `TpModel::Reset()` reaches it through every rank's `Model::Reset()` (`tp_model.cpp:784-796`).
  Tickets self-reset on every completed launch, so this only matters after a kernel that did not
  complete (a device fault or TDR); a host-side exception or a TP abort cannot leave one non-zero,
  because an enqueued GEMM always runs to completion.
  `ws` comes from the per-layer arena, and every slot is written before it is read.
- **The split tail is not free.** A split 128-group adds a serialized tail per linear: partial
  store, release fence, an L2 atomic with return, acquire, SKG loads, FWHT, store. That is about
  three dependent L2 round trips, estimated at 0.5-1.5 µs per linear, over up to about 330 linears
  per token: 0.6-1.8% of the step, which is more than A3's whole margin. It is unmeasured; M1
  measures µs per linear at SKG ∈ {1, 2, 4, 8} and `Wc` ∈ {32, 64, 128} on an N = 5120 row-parallel
  shape with the full epilogue, and the tuning sweep prefers in-WG SK where it ties.
- **Launch count.** There is one launch per linear, with no host sync and no inter-WG spin, so the
  kernel is safe under HIP-graph capture. Production captures nothing today (`docs/tp.md:38`).
- **Precision.** fp32 all the way from the accumulator through the FWHT and svh, then a single bf16
  rounding. This is better than an output transform in the consumer, which would round twice.
- **Row-parallel linears under TP.** Each rank applies the transform to its partial before
  `AllReduceSumBf16`. The math is exact because `P_n` and svh are linear. The rounding matches
  today's bf16-partials all-reduce, and no residual kernel changes.

### 4.6 Performance model and what it means for the gates

**Cost of one 256-weight fragment on one SIMD (M = 1..16).** Each lane owns 8 of its weights.

- **Decode:** 31 VALU cycles at KB = 4 (3.875 per weight; 37.5 at KB = 5), plus the loop and
  address overhead: 2 cycles at the 0.25 op/w that scalar addressing should reach, 5.5 at the
  probe's measured 0.69 (4.2).
- **WMMA:** 16 cycles, the same for any M ≤ 16; it stays at M = 1 because of the row-identity rule.
  Whether it overlaps VALU on gfx1201 is unknown. C37 (47.8 TF/s fp32 vector × 4 = the 191 TF/s
  WMMA peak) is consistent with WMMA running through the same SIMD datapath, so **no overlap is the
  base case**.
- **Memory:** 128.1 B at KB = 4. With the effective bandwidth BW shared by 128 SIMDs, the budget is
  `128.1 / (BW/128) × SCLK` cycles: 79, 64 and 54 cycles at 604 GB/s (C5) and 2.92, 2.35 and
  2.0 GHz; 87, 70 and 60 at the 550 GB/s the live gate_up measured (C8).

With no overlap and 0.25 overhead, need/budget is 0.62, 0.77 and 0.90 at those clocks (604 GB/s);
w4a16's is about 0.35. Trellis stays under the memory time in every case, so the question is how
much of its ALU time is left exposed. The model charges each fragment
`max(need, mem) + β·min(need, mem)`, where β is the exposed fraction of the shorter pipe (β = 0 is
perfect hiding).

**The baseline is per shape, at production groups** (q2ab_hv2_q3, `docs/quant2.md:1190-1192`: g128
on gate_up L32-63 and down L0-31; g32 on down L32-63, attn.o, and z/out_proj L0-31; bf16 attn.k/v
L32-63). The first revision compared against uniform g64. Trellis/today byte ratios:

| gate_up | down | qkv, qg | z, out_proj | o | k, v | all linears |
|---|---|---|---|---|---|---|
| 0.915 | 0.866 | 0.890 | 0.843 | 0.801 | 0.39 | **0.880** |

gate_up carries about 43% of the linear time and has the least headroom: against its g128 half,
break-even needs 0.942 of w4a16's effective bandwidth.

**Model** (byte rooflines only, `docs/r9700.md` rule 2; the tuning table's times back out above DRAM
peak, C8, and are used for nothing here). Today's linears are taken as memory-bound at BW: 22.9 ms
of the 27.86 ms step at 604 GB/s, 25.2 ms at 550. Plain decode change for `K4m` (604 / 550 GB/s):

| SCLK | WMMA/VALU overlap | β = 0 | β = 0.2, overhead 0.25 op/w | β = 0.2, overhead 0.69 op/w |
|---|---|---|---|---|
| 2.92 GHz | yes | +11% / +12% | +4.0% / +5.1% | +3.3% / +4.4% |
| 2.92 GHz | no | +11% / +12% | +1.0% / +2.0% | +0.3% / +1.3% |
| 2.35 GHz | yes | +11% / +12% | +2.5% / +3.5% | +1.6% / +2.7% |
| 2.35 GHz | no | +11% / +12% | **-1.2% / -0.2%** | **-2.0% / -1.0%** |
| 2.0 GHz | no | +11% / +12% | **-3.1% / -2.1%** | **-3.9% / -3.0%** |

The exposed fraction at which trellis meets A3 exactly (-0.5%) is β ≈ 0.15-0.21 with no overlap
at 2.0-2.35 GHz, 0.22-0.25 at boost, and 0.21-0.38 with overlap. The first revision's lane-rate
model gave -3.2% "central" (-5.6% with the probe's overhead, as the review reproduced); that is the
no-overlap, β = 0.2 case at an effective clock near 2.1 GHz on cache-flattered memory times.

**The expected case fails A3.** The base case (no overlap, a clock sagged to about 2.35 GHz, β
about 0.2) gives -0.2% to -2.0%, and -2% to -4% at 2.0 GHz. The first revision called this "a tie";
under A3's -0.5% bar it is a fail. Trellis passes only if its ALU time hides behind memory to within
β of about 0.15-0.2, or if WMMA overlaps VALU. Only M1 can say which.

The model is `tools/reference/trellis_perf_model.py` (committed in M0; CPU only).

**Clock.** Near the ridge, trellis time scales with 1/SCLK while memory-bound w4a16 does not. The
sustained clock during decode is unknown (Q8 is open; C38 documents a 2.2× boost ramp), an isolated
warmed-up microbenchmark probably runs at boost, and a VALU-heavy kernel draws more power under the
300 W TBP. So every M1 kernel records SCLK itself. A device-only compile (never launched) shows
`clock64()` lowering on gfx1201 to `s_getreg_b32 hwreg(HW_REG_SHADER_CYCLES_HI/LO)` (a hi-lo-hi
read) and `wall_clock64()` to `s_sendmsg_rtn_b64 MSG_RTN_GET_REALTIME`; the ratio of their deltas
times the wall-clock rate (99.1-99.7 MHz, as the TP code measured it) is the shader clock.

**Not modelled, on both sides.**

- Against trellis: the split-group tail (4.5, estimated 0.6-1.8% of the step); grid quantization
  at 40-80 WGs (4.3); the input transform, which launches fewer waves than the cast it replaces
  (4.8); LDS occupancy at large SK; and the clock sag its own ALU load causes.
- For trellis: `AFFINE` (4 cycles per fragment less); and M5's fused producers, 194 fewer launches
  per token than q2ab_hv2_q3 (5.4). The first revision put +1.5-3% on the launches, which has no
  support: the last launch-count cut measured flat because decode is GPU-bound
  (`docs/status.md:2365-2376`), and the fused w4a16 cast measured -4.3% (`linear.cpp:220-228`).
  No number until M5 measures it.

**DFlash and MTP** are predicted from their own rounds, not from plain decode. The w4a16 drafter and
the MTP head are unaffected, which dilutes any change on the linears. At M = 8 or 4 the GEMM costs
what it costs at M = 1, but the input transform, the FWHT epilogue and the workspace scale with M.
The sign tracks plain decode; the size does not. The round breakdown in `docs/tp.md:111` (36.8 ms
with a 6.3 ms drafter) predates today's 108 tok/s DFlash, so M1 re-measures the round composition
rather than reusing it.

**What this means.**

1. **KB = 4 is the only point with speed upside.** The mix streams under 1% fewer bytes than today
   (13.55 vs 13.68 GiB) and pays KB = 5's extra ALU (37.5 cycles and 160 B per fragment), so it
   loses A3 unless its ALU hides completely. `K4m` is the ship point; the mix is a quality-only
   fallback that must still pass A3, which in practice means D1 decides its fate.
2. **A3 depends on numbers only a GPU can give:** the issue rates of `v_mad_u32_u16`,
   `v_pk_mad_u16` and `v_sad_hi_u8` (all assumed full-rate; `docs/trellis.md` Q13); whether WMMA and
   VALU overlap; the exposed fraction β at 1, 2 and 4 waves per SIMD; the clock in the model; and the
   split tail. M1 measures all of them before any integration.
3. **One budget for M1 and M2.** A3 allows trellis's linear chain to be at most **0.14 ms per
   token slower** than q2ab's (0.5% of the 27.86 ms step). With S = M1's raw-GEMM saving against
   q2ab's GEMMs and X = M2's added work (the input transform beyond q2ab's casts, GDN Hadamards and
   residual rotations, plus the FWHT epilogue and the split tails), A3 needs `X − S ≤ 0.14 ms`.
   Reserving X ≤ 0.30 ms for M2 gives:
   - M1's raw GEMMs (no transforms) must save **S ≥ 0.16 ms per token** against q2ab's GEMMs at
     production groups: a weighted effective-bandwidth ratio of about **0.886** of w4a16's (0.875
     is break-even with X = 0; the byte ratio is 0.880);
   - M2's additions must stay at **X ≤ 0.30 ms per token**, and `X − S ≤ 0.14 ms` must still hold
     with the measured values.
4. **Levers, in order:**
   - `AFFINE = 1`: the B fragment is the exact f16 `1024 + sum`, and the epilogue applies
     `α·acc + β·Σa` per row. Since `H(1) = √128·e_0`, the β term touches only column 0 of each
     128-group. It saves 0.5 op/w (4 cycles per fragment). It is not bit-exact to the codebook (max
     9.77e-4, about 1e-5 of the quantization MSE), so it needs A2 re-checked.
   - `Wc`/`SK`/`SKG` shape (fewer split tails, better grids).
   - `NT`.
   - `U`.
   - **Per-shape coverage (a mixed body).** Keep w4a16 where trellis saves the fewest bytes
     (gate_up L32-63 at g128, 0.942; down L0-31 at g128) and use trellis where the baseline is g32 or
     bf16 (attn.o 0.80, the g32 halves of z/out_proj, attn.k/v). The catch: trellis excludes the
     q2ab rotation, so the kept w4a16 tensors would be unrotated and lose the rotation's KL gain. This
     needs a converter flag, relaxing 3.3's coverage rule, and its own KL run, so it is considered
     only after M5 and only if A3 misses.

### 4.7 Prefill

- **Path.** The same kernel with MT = 4 runs 64-row chunks. `kMaxChunkM` = 64 (`linear.cpp:17`);
  `max_chunk_` (`model.cpp:310`).
- **Cost.** Each decoded fragment feeds MT WMMAs, so decode is amortised 4×, but the M = 64 GEMM is
  WMMA-bound and the decode adds to it. w4a16's unpack is 1.25 op/w and costs 11.5-18.0% of its
  M = 64 time (`r4d_gemm_w4a16_nt_m64.hip:123-127`), i.e. 9.2-14.4% per op/w; the first revision
  used "about 8%" from `:20-27`, the low end. Trellis adds about 2.9-3.3 op/w more (3.875 plus
  overhead, against 1.25), so its M = 64 GEMM is about **+29-45%** against w4a16's, before the
  activation re-read: at NP = 1 one A fragment feeds 2 column tiles, while w4a16 uses NPW = 4 at
  M = 64 on 6 of its 10 shapes, and that file's header calls the re-read the M = 64 bottleneck.
  GEMMs are 67-79% of a 64-token chunk (66.8-73.2% measured at T ≤ 64 on an older container,
  `docs/r9700.md:500-507`; about 79% from table sums, which rule 2 allows only as a ratio), so
  prefill is expected at **0.74-0.84× of q2ab** (about 740-840 tok/s against about 1000). A3p
  (0.80×) sits inside that range and the activation re-read pushes toward its low end, so M8 is
  planned work.
- **M1 already measures the prefill shape:** (MT, NP) ∈ {(4, 1), (2, 2), (4, 2) at U = 1} at M = 64
  on gate_up and down. (4, 2) is about 195 VGPRs (4.4).
- **No reconstruct path.** There is no crossover in chunk-major prefill. The only dense GEMM,
  `r4d_gemm_bf16_nt_m64`, is already slower at M = 64 on gate_up (697.8 µs in the table, against a
  projected 460-530 µs for trellis) before any reconstruct cost; a dense copy would be rebuilt every
  chunk because r4dx's prefill is chunk-major; and bf16 cannot hold the f16 codebook values or the
  f16 A exactly. EXL3's 144-row switch (`docs/trellis.md:584-589`) does not carry over. Revisit only
  if r4dx gains a layer-major, large-M f16 GEMM.
- **M8: tiled prefill.** A kernel with Mt = 128 rows per WG: waves decode the WG's `Nt × Kstep`
  tiles once into LDS as f16, and the row-tile waves reuse them, which brings decode down to about a
  third of the WMMA time. This is the roadmap P9 kernel (`docs/perf.md:687-695`) with trellis decode,
  and the same kernel would help w4a16. **It is LDS-bandwidth-bound as first sketched:** 8 row-tile
  waves each reading a 512 B B-fragment is about 4.6 KB of LDS traffic per fragment, about 110
  CU-cycles at C12's 42 B/clk/CU against about 64 CU-cycles of WMMA. So each wave must own MT ≥ 2
  row tiles (halving the LDS reads), and M8's design waits for Q12 (the real LDS ceiling; C12 is
  probably an underestimate). No speed-up figure is quoted until then.

### 4.8 FWHT-128 and the input-transform kernels

**`Fwht128Wave`** (`r4d_fwht128_wave` in libr4d's `r4d_fwht128.h`). There is one definition, no
copy: `src/kernels/src/trellis_transform.hip` includes the libr4d header (through libr4d's `-I`), so
the input transform and the GEMM's output transform run the same butterfly. `test_trellis_input`
pins it bitwise to `FwhtLds` through the `r4dx_fwht128_f32` diagnostic.

- One wave handles 128 points. Lane l holds `i = l + 32r`, r = 0..3.
- Stages `lg = 0..4` are cross-lane: the partner is `l ^ (1<<lg)`, fetched with DPP `row_xmask` for
  lg ≤ 3 and `v_permlanex16` for lg = 4.
  - The lower lane computes `a + b`; the upper lane computes `partner - own`.
- Stages `lg = 5, 6` are in-register over r.
- The fp32 ops and their order equal `FwhtLds` (`hadamard_device.h:35-49`, lg = 0..6), so the
  existing fp64 reference (`tests/kernels/rotation_ref.hpp`) and the LDS kernels agree bit for bit
  in fp32.
- Cost is about 48 VALU per 128 points: negligible.

**`r4dx_trellis_input_bf16(x, M, K, nout, suh, out, prescale_log2, stream)`** (new
`src/kernels/src/trellis_transform.hip`, declared in `kernels.h`). `suh` and `out` are **host**
arrays of `nout` device pointers (`const int64_t*`); there is no `part_stride` argument (as built in
M2).

- Grid `(K/128, M)`, one wave (32 threads) per WG, one 128-block each. So K only needs to be
  ÷128, the loader's own rule (the first revision's `K/512` grid needed ÷512; every shape passes
  either way, including rank K 8704 and 3072). At decode, K = 5120 gives 40 WGs of 1 wave. Today's
  cast is 20 WGs × 8 waves (`model_kernels.hip:15`, `:31-37`), so this is more WGs but fewer waves,
  and it does more work per element; the first revision's "wider grid than today's cast" was wrong.
  Its cost counts against M2's 0.30 ms budget (4.6).
- It reads one bf16 128-block of x, then for each output `o < nout` (up to 3):
  `v = x·suh_o` (fp32, suh widened at load), `Fwht128Wave`, then
  `out_o = f16_rn(v · 2^s/√128)`. Output o is written at `out[o] + row·K`. For one linear's parts
  the caller passes `suh[1] = suh[0] + K` and `out[1] = out[0] + part_stride` (5.3).
- The product `v · 2^s/√128` is rounded to fp32 before the f16 conversion. Left alone, the compiler
  fuses the multiply and the conversion into one `v_fma_mixlo_f16`, which rounds the exact product
  once and differs in the last f16 bit on about 1e-5 of the elements; an empty `asm` keeps the fp32
  product (M2). The M5 fused producers must do the same, or they are not byte-identical to v1.
- **There is exactly one rounding, straight to f16.** The q2ab kernels round to bf16 first and then
  to f16 (`r4dx_kernels.hip:499-501`); the trellis variants must not.

**Fused producer variants (M5).** Each adds an `f16_out` form, and each is **byte-identical to v1**
(the producer's own bf16 output, then `r4dx_trellis_input_bf16`). The q2ab kernels feed the fp32
product straight into the transform (`SiluMulHadamardBlockKernel`, `r4dx_kernels.hip:490`); the
trellis forms instead round the producer value to bf16 first, exactly as v1's separate
`r4dx_silu_mul_bf16` does, and then multiply by suh in fp32. Otherwise the shipped M5 binary would
not be the binary A1 and A2 measured.

| Variant | Based on | Change |
|---|---|---|
| `r4dx_silu_mul_trellis_bf16` | `SiluMulHadamardBlockKernel` (`r4dx_kernels.hip:477-503`) | block 128; `bf16_rn(silu(g)·u)` first, then `·suh_down` (fp32); f16 A out |
| attention gate-mul trellis form | `r4dx_model_attn_gate_mul_hadamard_bf16` (`attn_kernels.hip:52-73`, `:98-127`) | block 128; bf16 product first, then `·suh_o`; f16 out |
| GDN out_proj | `r4dx_trellis_input_bf16` with `nout = 1` on `out_core` | the same kernel as v1; replaces q2ab's in-place Hb and the cast (`gdn_layer.cpp:236-252`) |

`R4DX_DISABLE_EPILOGUE=1` also turns the trellis fusions off (one helper, `TrellisFusionEnabled()`,
next to `EpilogueForLayout`), so a single binary gives the fused/unfused SHA A/B that
`tools/validate_fusion.ps1` runs today for w4a8/mxfp4; M5 adds `trellis` to its `-Layouts`.

**Prescale `s`.** This is a power of two applied to A and undone in the epilogue: exact, and free.
It exists in case `x⊙suh` falls into the f16 subnormal range. suh is about |W|/(1.24·g), roughly
1e-2. M4 measures `max|A|` and the subnormal fraction on the KL corpus and picks `s`. The default is
0. It lives in the container (`quant.trellis.prescale_log2`, 2.3) and in
`QuantLinear::trellis_prescale_log2`; the transform takes it as an argument and the GEMM receives
`out_scale = 2^-s/√128`.

### 4.9 Fused pairs

| Container linear(s) | Shared input | Handling |
|---|---|---|
| `mlp.gate_up` (gate and up, **different suh**, `trellis_quant.py:1081`, `:1092`) | `x_normed` | One QuantLinear with 2 parts. One input transform with `nout = 2`, then **one GEMM**, where A is chosen by part per WG (`n_split` = I; 17408 and 8704 are ÷256). The output layout is `[gate \| up]`, so `silu_mul` is unchanged (`mlp.cpp:103-122`). |
| `attn.qg`, `attn.k`, `attn.v` | `normed` | Three QuantLinears. One input transform with `nout = 3` (M5), then three GEMMs. `qg`'s per-head `[q\|gate]` interleave holds 4 whole 128-blocks per head, so `split_qg` is unchanged. |
| `gdn.in_proj_qkv`, `gdn.in_proj_z` | `x_normed` | Two QuantLinears. One input transform with `nout = 2` (M5), then two GEMMs. `in_proj_a/b` stay bf16 on the raw `x_normed` (`gdn_layer.cpp:114-122`). |
| `mlp.down`, `attn.o`, `gdn.out_proj` | producer output | The producer's fused variant (4.8). |

**Later options, not in v1:**

- **Fold SwiGLU into the gate_up epilogue.** The converter interleaves gate and up by 128-blocks,
  so the finisher WG for block j computes FWHT·svh on g and u, then `silu(g)·u·suh_down`, then
  FWHT again, and writes f16 A_down. That saves one launch and one bf16 round-trip per layer. It
  changes `mlp.cpp` and the TP row rule, so it waits until after A3.
- **A grouped k+v launch**, which saves 16 launches per token.

### 4.10 Rejected alternatives

| Alternative | Reason |
|---|---|
| dot2 GEMV at M = 1 with WMMA for verify | Breaks verify == decode bit-identity (`linear.cpp:19-33`). Using dot2 everywhere costs 4 op/w at M = 8 (DFlash about -20%). |
| int8-activation GEMV (EXL3 G8) | Its integer sums would be order-independent, but it costs 8 dot4 per weight at M = 8, adds activation error, and does not use the fp16 codebook. |
| Re-encoding in a WMMA-native order | The column split already reaches 0.375 op/w extraction. It would also mean re-running the oracle. |
| Output transform in consumer kernels | Two roundings, and conv/norm/rope/kv_write would all change. |
| Grid-sync input prologue inside the GEMM | HIP needs a cooperative launch for that. The per-WG redundant alternative costs about 20% at M = 16. |
| LUT codebooks | 128 KiB exceeds a block's LDS, and a gather costs about what it replaces (perf survey 5.9). |
| Shared suh per input group | The bits would have to be re-quantized, and the quality cost was never measured. |

---

## 5. Runtime integration

### 5.1 Types and loader

| File:line | Change |
|---|---|
| `quant_linear.h:18` | Append `kTrellis` to `Layout`, **last**, so the enum values keyed into the tuning cache do not move (`linear.cpp:175-176`). |
| `quant_linear.h:76-107` | Add to `QuantLinear`: `DeviceBuffer<uint32_t> trellis_w`; `DeviceBuffer<float> trellis_suh` (`[P][K]`, widened from fp16); `DeviceBuffer<float> trellis_svh` (`[N]`); `uint32_t* trellis_tickets` (`[N/128]`, a slice of the container's one ticket buffer, 4.5); `int trellis_bits`; `int trellis_parts`; `int64_t trellis_part_n[2]` (rank-local); `int trellis_prescale_log2`. |
| `quant_linear.cpp:7-23` | `"trellis"` in `LayoutName` and `LayoutFromName`. This reaches every `--layout` parser (cli, server, `tool_teacher_forced_logprobs`). Their usage strings (`cli_args.h:226`, `server_args.h:139`, `tests/model/tool_teacher_forced_logprobs.cpp:23`) gain `trellis`. |
| `container.cpp:87-103` | `CheckQuantGroups` keeps gating on any w4a16 layout: lm_head and MTP are still w4a16. |
| `container.cpp:239-283` | `LoadQuantLinear` / `HasLayout` gain a `kTrellis` case: 3 tensors, the size checks in 2.5, widening. |
| `container.cpp:312-338` | No change: tier 2 serves `--keep-bf16` bases. |
| `container.cpp:439-499` | **Head layouts:** at the top of `Container::Load(path, o)` (so both the TP = 1 path and `LoadShard` see it), a requested `lm_head_layout` or `mtp_head_layout` of `kTrellis` is mapped to `kW4a16`. Every caller then gets the same answer, including tests that call `Container::Load` directly with the body layout, and `Model::Load` (`model.cpp:221-228`, which passes the body layout for lm_head) needs no new option. New `ParseTrellisMetadata`, the refusals in 2.5, and the rotation exclusion. |
| `container.cpp:1002-1036` | `LoadShard` calls `ParseTrellisMetadata` next to its own `ParseRotationMetadata` (`:1031`), applies the same refusals, and passes the spec to `ShardLoader`. |
| `container.cpp:829-921` | `ShardLoader::Linear` gains a `kTrellis` case with **explicit** plans; the generic `Part()` helper (`:937-952`) cannot serve suh/svh, because it picks the plan by the linear's rule: under `kRows` it would cut suh by N rows, and under `kCols` `PlanCols` refuses an elem part (`tp_shard.cpp:330`). So: `.trellis.w` through `Part()` with the new `kTrellisW` part; `suh` replicated under `kRows`, and `PlanRows(kElem, row_bytes 2, [k0, kc))` over the rank's K range under `kCols`; `svh` by `PlanRows(kElem, rows)` under `kRows`, replicated under `kCols`. Both are gathered as fp16 and widened on the host before upload (there is no gather-then-widen helper; `WidenedF32`, `:815-824`, is replicated-only). Plus rank-local `part_n`, and the 128-alignment checks of 2.5. |
| `tp_shard.cpp` `PartBytes`, `PartShape` | `kTrellisW` bytes are `N·K·KB/8`; `PartShape` carries KB (in its `group` field or a new one). |
| `dflash_draft.cpp:142-159` | `case kTrellis: throw` (the drafter is never trellis). |

### 5.2 Model

| File:line | Change |
|---|---|
| `model.cpp:221-228` | No change: `Container::Load` maps the trellis head layouts (5.1). |
| `model.cpp:331` | No change: `EpilogueForLayout(kTrellis)` is `r4dx_epilogue_none` (5.3), so `body_epilogue_` is none. rmsnorm never fuses the transform: a one-WG-per-row shape loses parallelism, the -4.3% measured at `linear.cpp:220-228`. |
| `model.cpp:576-638` | `Model::Reset()` calls `container_.ZeroTrellisTickets(stream_)` before its existing `stream_.Synchronize()` (4.5). |

### 5.3 Linear dispatch (`linear.h`, `linear.cpp`)

| File:line | Change |
|---|---|
| `linear.h:17-19` | `LinearTuning` gains `int SKG = 1` and `int U = 2`. For trellis, `MB` means MT and `NPW` means NP. Both new fields are ignored by the other layouts, whose table rows keep their current five values. |
| `linear.h:40-45` | `GemmTuningRow` gains `int rate = 0` (KB for trellis). |
| `linear.h:59` | `PickTuning`'s last argument becomes a generic "variant": the w4a16 group, or KB for trellis. The cache key keeps bits 53+ for it (`linear.cpp:175-176`). |
| `linear.cpp:56-73` | `FallbackTuning` gains an `M` argument (`ResolveTuning` already has it) and the trellis rule from 4.4, M-aware; legality `(K/16) % (SK·SKG·U)`. |
| `linear.cpp:109-124` | `BestRow` matches `rate` for trellis. |
| `linear.cpp:141-145` | The M ≤ 16 → M = 1 band rule covers trellis too (SK, SKG, WV, NP, MT, U). NT per band. |
| `linear.cpp:184-250` | `EpilogueForLayout(kTrellis)` returns **`r4dx_epilogue_none`**. The layers call `EpilogueForLayout` themselves (`gdn_layer.cpp:74`, `:133`; `mlp.cpp:52`, `:91`; `attention_layer.hpp:139`, `:200-201`) and would hand any other value to `r4dx_rmsnorm_bf16`/`r4dx_silu_mul_bf16`, whose `ApplyEpilogueRow` (`r4dx_kernels.hip:105-159`) silently writes nothing for an unknown value, leaving the `pre` buffer uninitialized. With none, every layer runs its plain bf16 path in v1. The M5 fusions never go through this function (5.4). |
| `r4dx_kernels.hip:163` and the rmsnorm / residual_rmsnorm / silu_mul host entries | `CheckEpiloguePrecondition` throws on any epilogue value it does not know, so an unknown value can never again reach `ApplyEpilogueRow`. Shared code: covered by A6. |
| `linear.h:89-93` | `PreQuantizedActivation` gains `const void* transform_id` (must equal `w.trellis_suh.data()`) and `int64_t part_stride` (elements between parts). For trellis, `data` holds `P` parts, each `[M][K]` f16, part `p` at `data + p·part_stride`; chunk `m0` is at `+ m0·K` inside each part (`linear.cpp:318` offsets by `m0*K` only, which is right within a part and wrong across parts once M > 64 without the stride). `ApplyLinear` takes the trellis path when `pre && pre->transform_id != nullptr`, and throws if the id is not this linear's: the transform is per linear, so an epilogue-kind check is not enough (`linear.cpp:262-267`). |
| `linear.cpp:269-298` | Scratch: without `pre`, `P·kMaxChunkM·K` f16 for A, with `part_stride = kMaxChunkM·K`. For split 128-groups at M ≤ 16, `r4d_gemm_trellis_nt_m64_ws_bytes` of fp32 from the arena: at most 16·34816·8·4 B = 17.8 MB, inside the 96 MiB arena (`model.cpp:345`). |
| `linear.cpp:310-376` | New case, per chunk. Without `pre`: `r4dx_trellis_input_bf16(xc, m, K, P, suh_ptrs, out_ptrs, s, stream)` with host arrays `suh_ptrs = {suh, suh + K}` and `out_ptrs = {scratch, scratch + part_stride}` (4.8; the kernel takes no stride). Then `GemmTrellisNtM64(a0, a1, part_n[0], ...)` with `out_scale = 2^-s/√128`. A Wc = 32 table row (out_proj/o, k/v, down) takes the split path, so it needs `ws` even at SKG = 1. |

**v1 is self-contained.** Because `EpilogueForLayout(kTrellis)` is none, every existing call site
already passes plain bf16 activations and no `pre`, and becomes correct as soon as the loader
accepts the layout. Launches per token are **34 fewer than q2ab_hv2_q3's**. The transform replaces
the per-linear cast one for one, but there are 336 transforms against 320 casts, because q2ab keeps
attn.k/v bf16 in layers 32-63 (16 linears that need no cast). Against that, on an unrotated
container the 48 GDN in-place Hadamards (`gdn_layer.cpp:241-247`; `HadSigns()` returns nulls,
`model.cpp:137-145`) and the two residual rotations (`RotateResidual`, `model.cpp:128-135`)
disappear.

### 5.4 Layer code (M5, fused producers; v1 needs none)

| File:line | Change |
|---|---|
| `attention_layer.hpp:172-207` | When `qg`, `k` and `v` are trellis: one `r4dx_trellis_input_bf16` with `nout = 3`, then three `pre`s with their `transform_id`s. |
| `attention_layer.hpp:299-322` | The trellis gate-mul variant feeds `o`'s `pre`. |
| `gdn_layer.cpp:101-137` | One transform with `nout = 2` for `in_proj_qkv` and `in_proj_z`. |
| `gdn_layer.cpp:236-252` | `r4dx_trellis_input_bf16(out_core)` feeds `out_proj`'s `pre`. |
| `mlp.cpp:103-129` | `r4dx_silu_mul_trellis_bf16` feeds `down`'s `pre`. |
| `mlp.cpp:77-82` | Unchanged: `ApplyLinear` already makes one transform launch for both parts. |

Each fused path is selected by `TrellisFusionEnabled()` (off under `R4DX_DISABLE_EPILOGUE=1`) and
passes a `pre` with `transform_id` set; `EpilogueForLayout` stays none. The fused kernels are
byte-identical to v1 (4.8), which `test_trellis_input` checks per kernel and M5's
`validate_fusion.ps1 -Layouts trellis` checks end to end (generated-text SHA, fusion on vs off).

Net effect per token: 160 fewer launches than v1 and **194 fewer than q2ab_hv2_q3** (the first
revision's 208 assumed quantized k/v in every layer). Against q2ab: GDN+MLP -3 per layer (48),
attention+MLP -4 per layer in layers 0-31 (8) and -2 in layers 32-63 (8), and the 2 residual
rotations; at 1.2 µs of host time plus a few µs of GPU time each (`docs/r9700.md:70`). No speed is
claimed for it until measured (4.6).

**Output side.** No layer changes: C is already in the original basis.

### 5.5 Tensor parallelism

| File:line | Change |
|---|---|
| `tp_shard.cpp:18-25` | `HasLayoutSuffix` accepts `"trellis"`. |
| `tp/tp_shard.h` Part enum; `PlanRows` / `PlanCols` (`tp_shard.cpp:286+`) | New `Part::kTrellisW` with the pair-grid runs from 2.4 and a 128-granular range check. suh and svh use the existing element part (2-byte rows) through the explicit plans in 5.1. |
| `model_config.h:239-252` | No change. `ModelConfig::Shard` has no container or layout knowledge; the trellis ÷128 checks on rank segments and rank K live in `LoadShard`/`ShardLoader` and `kTrellisW`'s plans (2.5). Rank K is already required ÷512 there (`:239-242`). All pass at TP = 2. |
| `gemm_tuning_table_tp2.inc` | Per-rank trellis rows. |

The epilogue runs before `AllReduceSumBf16` (`mlp.cpp:132-135`, `gdn_layer.cpp:256-258`,
`attention_layer.hpp:327-331`), so those call sites are unchanged. The MTP head stays w4a16 and is
sharded as today (`tp_shard.cpp:222-234`).

### 5.6 Workspace, graph capture, tuning

- **Workspace.**
  - Arena scratch covers A-parts and split-group partials.
  - Tickets live in one persistent buffer per `Container` (so per TP rank), zeroed at load and by
    `Model::Reset()` (4.5). There is no module-scope scratch, which would be unsafe with two rank
    threads (`kernels.h:341-370` precedent).
- **Graph capture.** No host sync, no allocation beyond arena bumps, and self-resetting counters: the
  kernel is capture-safe.
- **Tuning** (`tools/profile/tune_gemm.py`).
  - Add a trellis timing branch and `LAYOUT_ENUM` entry, and emit `rate`.
  - Sweep M ∈ {1, 32, 64}, plus M = 8 for NT only, over the knob space in 4.4. The M = 1 sweep
    is timed on a pooled working set larger than the 64 MiB MALL and ranks configurations only
    (`docs/r9700.md` rule 2); speed claims come from `bench_decode.ps1`.
  - Random words decode to valid values, so no real weights are needed.

---

## 6. Tests

**Golden vectors.** `tools/reference/trellis_golden.py` is CPU only. It uses numpy plus
`trellis_quant.py`'s CPU functions, never `torch.cuda`. It writes `tests/kernels/golden/trellis/*.npy`
(small; committed with a sha manifest), using `npy_fixture.hpp` for reading:

- random tiles at KB = 4 and 5 in the pair grid;
- their `decode_words` f16 Q;
- random A and suh, the fp32-emulated transform outputs (same butterfly order) and the fp64 full
  linear;
- **real tiles** when `D:\models\r4dx\trellis-q\K4m` exists: a 256×256 block of L03 `attn.k`, L10
  `mlp.down` and L07 `mlp.gate_up` (gate and up), cut from the oracle files with their `suh`/`svh`
  slices.

**Tests** (GPU tests run on device 1 via the CMake property; `/EHc-` for precondition throws, as in
`tests/kernels/CMakeLists.txt:45-56`, `:90-92`):

| Test | Where | Checks |
|---|---|---|
| `test_trellis_decode` | tests/kernels | `r4d_trellis_reconstruct_f16` is **bit-exact** against `decode_words`: random and real tiles, KB = 4 and 5. `_raw` GEMM with one-hot A rows returns decoded weights **bit-exactly** for every tuning row of both tables (tests the lane map, pair grid, SK/SKG reduction and part selection). |
| `test_trellis_gemm` | tests/kernels | Full linear (transform + GEMM + epilogue) against fp64 on random and real tiles, M ∈ {1, 3, 8, 16, 17, 64}, P ∈ {1, 2}. Tolerance: per element ≤ 4 bf16 ulp of the fp64 value computed from the f16-rounded A, and relative Frobenius ≤ 2e-3. **Row identity:** for every M ≤ 16 tuning row, row r of an M-row call is bit-identical to the M = 1 call on the same input. Determinism: 100 repeats at SKG = 8 and at `Wc` = 32 give identical bytes. Ticket reset: with tickets deliberately left non-zero the result is wrong (so the test can see the failure), and after `ZeroTrellisTickets` the next call is bit-exact again. Precondition throws (including `NP·U > 8` and the LDS rule). |
| `test_trellis_input` | tests/kernels | `r4dx_trellis_input_bf16` is **bit-exact** against the fp32 emulation: nout 1-3, prescale, outputs at a part stride, M = 1..64, K ∈ {3072, 5120, 6144, 8704, 17408}. Each fused variant (M5) is **byte-identical to the v1 pair** (producer's bf16 output, then `r4dx_trellis_input_bf16`). `Fwht128Wave` equals `FwhtLds` bitwise. |
| `convert_trellis_import` | tests/convert (CPU) | A synthetic 2-layer checkpoint plus a synthetic override directory written by the golden script (random words; checkpoint weights = their reconstruction, so verify passes exactly), in both manifest forms: a quantize-model manifest and a mix manifest (absolute `file` paths, no `stale_layers`/`layers_done`/top-level `code_sha256`, float `K`). Checks the regrid bytes against python, metadata and summary. Refusals: sha mismatch, `complete: false`, stale layer, K = 3.5, exl3 basis without the allow flag, `--rotate q2ab`, missing linear, gate K ≠ up K, config sha mismatch, `--trellis-from` with `--selftest`/`--dflash-gguf`/`--reuse-tensors-from`. Checks that `--keep-bf16` skips a manifest entry. Follows the `convert_rms_hessian` exe pattern (`tests/convert/CMakeLists.txt:64-71`). |
| `test_trellis_linear` + `test_tp_loader` cases | tests/model | Load the convert test's tiny container. `ApplyLinear` (with and without `pre`, P ∈ {1, 2}, M ∈ {1, 16, 17, 64, 65, 130} so the part stride and the chunk offset are both exercised) against CPU. `EpilogueForLayout(kTrellis) == r4dx_epilogue_none`. `Container::Load` with a trellis `lm_head_layout` loads w4a16. Loader refusals (2.5) in both `Load` and `LoadShard`. TP = 2 shard slices of every part, suh and svh included, equal the corresponding slices of the full buffers (`test_tp_loader.cpp` pattern). `test_pick_tuning`: trellis rows, and every pick for M = 1..64 on both the TP = 1 and TP = 2 tables and on the fallback is legal. |
| `test_fused_quant` case | tests/kernels | The rmsnorm, residual_rmsnorm and silu_mul host entries throw on an unknown epilogue value. |
| `test_trellis_gemm.py` | libr4d | The libr4d convention: random words against a torch-CPU reference through `r4d.pyd`. |

**End-to-end** (GPU, with approval; M4 and M6):

1. Convert `qwen38-27b-trellis-k4m.r4dx`, plus its `--lm-head bf16` twin for A2.
2. `tool_teacher_forced_logprobs --layout trellis --tokens tools\reference\kl_corpus\tokens_canon.json`,
   then `kl_report.py --ref-dir D:\models\r4dx\kl-canon\ref`. This gives A1.
3. `kl_report.py --ref-dir <oracle golden dump D:\models\r4dx\kl-trellis\K4m>` against the twin's
   runtime dump, as a diagnostic, together with A2's arithmetic on the two reference KLs.
4. Greedy sanity check: `r4dx-cli` on the haiku prompt.
5. **A6 (w4a16 regression)** on the final binary with q2ab_hv2_q3: full ctest, `g6_validate.ps1`
   (default layout), `tools/tp/tp1_identity.ps1`, and a generated-text SHA A/B against the
   pre-trellis quant2 binary.

**Benches** (GPU, with approval). `tool_trellis_gemm_bench` (tests/kernels; built, never
`add_test`'d) has five modes:

- **Linear-chain replay (M1's gate).** One decode step's 400 body linears in layer order, with
  random weights at full size in VRAM (trellis 11.3 GiB plus w4a16 at q2ab_hv2_q3's per-shape groups
  and its bf16 k/v, about 24 GiB together), timed back to back at M = 1 and M = 8, alternating the
  two formats in one process. The w4a16 side uses production's path: the M = 1 tuning for every
  M ≤ 16 and NT = 0 at M > 1 (`linear.cpp:137-145`), not the table's M = 8 row. It reports GEMM-only
  time per token for each format, per-shape µs and effective GB/s, and the in-kernel SCLK of both
  (WG 0 records `clock64()`/`wall_clock64()` at entry and exit).
- **Op rates and overlap.** Issue rates of `v_mad_u32_u16`, `v_pk_mad_u16`, `v_sad_u8`,
  `v_sad_hi_u8` and `v_mul_lo_u32`; and WMMA-only, VALU-only and mixed loops at 1, 2 and 4 waves
  per SIMD, which give WMMA/VALU overlap and the exposed fraction β directly.
- **Split tail.** µs per linear at SKG ∈ {1, 2, 4, 8} and `Wc` ∈ {32, 64, 128} on an N = 5120
  row-parallel shape (down), with the full epilogue once M2 has it (raw ticket plus workspace round
  trip in M1).
- **Prefill shape.** M = 64 on gate_up and down at (MT, NP) ∈ {(4, 1), (2, 2), (4, 2) at U = 1}
  against w4a16's M = 64 tuning.
- **In-model clock.** An `R4DX_CLOCK_PROBE=1` debug hook in r4dx launches a one-wave probe kernel
  (`clock64`/`wall_clock64` around a fixed VALU loop, about 5 µs) after every MLP during a real
  q2ab decode, so the replay's clock can be compared with the decode's. The same run records the
  DFlash and MTP round composition (drafter, verify, other) for 4.6's per-mode prediction.

Also: `bench_decode.ps1` gains a per-entry layout (`name=path[@exe][#layout]`), and a
`tune_gemm.py` sweep.

---

## 7. Milestones (in order, with go/no-go checks)

The effort figures are one engineer's working days. Every GPU step needs the user's approval, and
none can run while the oracle occupies GPU 1.

| # | Milestone | Effort | Go/no-go check |
|---|---|---|---|
| M0 | libr4d branch hygiene: push `99a5d94` to canonical, create branch `trellis`. Commit the lane-map check, `trellis_golden.py` and the perf model (`tools/reference/trellis_perf_model.py`). **The user decides D1** (section 1). | 0.5 d | A fresh clone resolves the pin. The lane check reports 0 mismatches. D1 recorded. |
| M1 | libr4d: `r4d_trellis_dq.h`, reconstruct, `_raw` GEMM (KB = 4, no output transform, raw ticket/workspace path for split groups, `Wc` ∈ {32, 64, 128, 256}, scalar addressing), `test_trellis_decode`; the five `tool_trellis_gemm_bench` modes (section 6). | 4 d | Whole-loop VALU count reported per instantiation. On the replay, at M = 1 and at M = 8: **GO** if trellis raw GEMM time per token ≤ q2ab's GEMM time − **0.16 ms** (S ≥ 0.16 ms, the 4.6 budget; a weighted effective-bandwidth ratio of about 0.886). **STOP and report** if trellis raw is slower than q2ab's GEMMs by more than A3's whole margin (S < −0.14 ms, a ratio below about 0.875), which no overhead budget can absorb. Between the two: `AFFINE`, `Wc`/SK/SKG, NT, U, then decide on the numbers with D1. If the replay's SCLK is more than 3% above the in-model clock probe's, the decision uses the trellis time rescaled to the in-model clock through the measured ALU-bound fraction. Also recorded: op rates, overlap and β, split-tail µs, prefill (MT, NP), DFlash/MTP round composition. |
| M2 | Epilogue (8-row LDS reduce, tickets, FWHT-128, svh, bf16), parts, KB = 5, `r4dx_trellis_input_bf16` (+ `src/kernels/CMakeLists.txt`), `test_trellis_gemm`, `test_trellis_input`. | 2.5 d | All tests green; row-identity and determinism pass. On the replay: X = (transform time − q2ab's cast, GDN Hb and rotation time) + epilogue + split tails **≤ 0.30 ms per token**, and `X − S ≤ 0.14 ms` with M1's measured S. |
| M3 | Converter import (section 3) and `convert_trellis_import`. Convert `K4m` and its twin on the CPU. | 2 d | `--trellis-verify full` passes for 400/400 linears. Decode bytes about 12.13 GiB (± 0.02), measured the way `docs/quant2.md` measured 13.68. |
| M4 | Runtime v1 (5.1-5.3, 5.5-5.6): loader (both paths), head-layout mapping, `EpilogueForLayout(kTrellis) = none`, epilogue-host throws, part stride, M-aware fallback tuning, ticket buffer and reset, TP slicing, `test_trellis_linear`/TP/`test_pick_tuning` cases; ctest green. KL runs (A1, A2) and the A-range study (4.8). | 3 d | **A2 passes** (else it is a kernel/format bug: stop and debug). **A1 passes** for `K4m`, or for `mix4.5m` after converting it; otherwise stop, no ship. Greedy output sane. |
| M5 | Tuning sweep (TP = 1 and TP = 2 tables), fused producers (5.4) with `validate_fusion.ps1 -Layouts trellis`, then `bench_decode.ps1` against q2ab_hv2_q3. | 2 d | Fusion SHA A/B byte-identical (so A2 carries over). **A3** on plain, DFlash and MTP. If it misses: `AFFINE` (then re-run A2); then D1 applies, and the mixed-body lever (4.6) is costed. A3p measured. |
| M6 | G6 on the candidate (`g6_validate.ps1 -Layout trellis`), A6 on q2ab_hv2_q3, docs (`container-format.md`, `quant2.md`, `perf.md`). | 1 d | **A4** 5/5, **A5**, **A6**. |
| M7 | Ship: production container and recipe recorded. | 0.5 d | All of A0-A6 (A3p via M8 if needed). |
| M8 | Tiled prefill with LDS-staged decode (4.7), started only after A3 passes, designed after Q12 is measured. | 3-4 d | Prefill ≥ 0.80× q2ab (A3p), and decode unchanged. |

**Critical path.** About 15.5 days without M8, and 18.5-19.5 with it, which the prefill estimate
(0.74-0.84×) makes the likely total. M0 and the CPU-only parts of M1-M3 (kernel code, converter and
golden script) can start before the oracle finishes, but nothing past A0 ships without it.

---

## 8. Risks and open questions

| # | Risk | Mitigation or trigger |
|---|---|---|
| R1 | **The decode's ALU time is not hidden.** The expected case (no WMMA/VALU overlap, about 2.35 GHz, 20% of the ALU time exposed) loses 0.2-2.0% and fails A3; at 2.0 GHz, 2-4%. Op rates are assumed full-rate. | M1 measures op rates, overlap, β and the clock before any integration; the M1 gate is derived from A3 (4.6). `AFFINE` saves 4 cycles per fragment. D1 decides in advance what a narrow miss means. |
| R2 | Prefill is 0.74-0.84× of q2ab at 64-row chunks. | M8 is planned; its design waits for Q12 (LDS ceiling). |
| R3 | The oracle's KL does not survive into the runtime: f16 A, bf16 C, w4a16 g32 lm_head, fp8 KV. | Gate A2 isolates the kernel error. The lm_head is measured separately (twin container). |
| R4 | f16 range of `P(x⊙suh)` (underflow for small activations, overflow on outliers; EXL3 does the same). | The M4 A-range study, and prescale `s`. |
| R5 | Ticket memory ordering on gfx12 (release/acquire at agent scope; L0 staleness). | 100-repeat determinism tests at SKG = 8 and at `Wc` = 32. An unsplit 128-group (SKG = 1, `Wc` ≥ 128) is always legal as a fallback. |
| R6 | KB = 5 register or VMEM pressure (5 b32 per tile per lane). | Only relevant to the mix, which is already a fallback. |
| R7 | The libr4d submodule pin cannot be resolved by a fresh checkout. | M0. |
| R8 | The mix's two rates came from different oracle code versions (`mix` only warns, `trellis_quant.py:1646-1649`). | The converter records both shas, and step 10's reconstruction check is authoritative. |
| R9 | Real TP = 2 (device 0 transport) is untested, the same as q2ab. | Emulate mode is the byte-exact reference for the sharded math. |
| R10 | **Clock.** Trellis time scales with 1/SCLK near the ridge; a microbenchmark at boost would pass a kernel that fails in the model. | In-kernel SCLK in every M1 kernel, the in-model clock probe, and the replay's alternating formats (section 6). |
| R11 | Split-group tails and grid quantization (40-80 WGs at `Wc` = 128). | `Wc` ∈ {32, 64}, in-WG SK up to 16 (8-row LDS passes), and M1's split-tail measurement, all inside M2's 0.30 ms budget. |
| R12 | The shared-code changes regress the w4a16 production path. | Gate A6. |
| R13 | A kernel that did not complete (device fault, TDR) leaves a ticket non-zero, and every later call of that linear is silently wrong. | One ticket buffer per container, zeroed by `Model::Reset()` (4.5). |

**Open questions that need a GPU (all answered by M1's benches unless noted):**

- **Q-g1.** Issue rates of `v_mad_u32_u16`, `v_pk_mad_u16`, `v_sad_u8` and `v_sad_hi_u8` on gfx1201
  (`docs/trellis.md` Q13). If `v_sad_hi_u8` is not full-rate, `v_dot4_u32_u8` plus `v_perm_b32`
  costs the same 3 ops per pair.
- **Q-g2.** Does WMMA overlap VALU, and what fraction β of the ALU time stays exposed at 1, 2 and 4
  waves per SIMD?
- **Q-g3.** The sustained shader clock in a real decode (`docs/r9700.md` Q8) against the
  microbenchmark's.
- **Q-g4.** µs per linear of a split 128-group at SKG ∈ {1, 2, 4, 8} and `Wc` ∈ {32, 64, 128}.
- **Q-g5.** The replay's per-shape effective bandwidth for trellis and for w4a16 at production
  groups, at M = 1 and 8 (the M1 gate).
- **Q-g6.** Input-transform cost against the cast it replaces (M2).
- **Q-g7.** M = 64 cost at (MT, NP) ∈ {(4, 1), (2, 2), (4, 2)}; and the LDS ceiling
  (`docs/r9700.md` Q12, a re-run of `lds_bandwidth.exe` before M8) that M8's design depends on.
- **Q-g8.** Today's DFlash and MTP round composition (drafter, verify, other) at 108 and 66 tok/s.
- **Q-g9.** The f16 range of `P(x⊙suh)` on the KL corpus, which picks `s` (M4).

**Open questions answerable without a GPU (none blocks M0-M3):**

- **D1** (a decision, section 1): what a wide A1 pass with a 1-4% A3 miss means. Needed before M1.
- **Q-b.** DPP `row_xmask` or `ds_swizzle` for the FWHT's cross-lane stages? Either is negligible;
  the choice is by ISA convenience.
- **Q-c.** Is fusing SwiGLU into the gate_up epilogue (4.9) worth a converter-side 128-interleave of
  gate/up? Decide after M5's per-shape profile.
- **Q-d.** Should `--trellis-verify full` also check the Hessian proxy? It would need the `.hess`
  files; `rel_weight_err` alone already catches regrid and read errors.
- **Q-e.** Is `mix4.5m` worth converting at all if `K4m` passes A1? It is not: about 13.55 GiB, and slower
  by the model. Convert it only if `K4m` fails A1.

---

## 9. Review log (revision 2)

Two adversarial reviews (performance: 15 findings; integration: 15 findings plus a list of claims
that held) were checked against the code on the CPU. All 30 were real in substance; none was
rejected outright. Where a finding's own evidence was off, the doc follows the code, as listed under
"Narrowed".

**Performance review, accepted:**

| # | Finding | Where fixed |
|---|---|---|
| P1 | "Central is a tie" was wrong: the central case fails A3. | 0, 4.6 ("The expected case fails A3"), R1, D1 |
| P2 | WMMA/VALU overlap carries the upside; model it as a per-SIMD cycle budget. | 4.6 rewritten; M1 overlap and β bench |
| P3 | Clock sensitivity; a boost-clock microbenchmark could pass a failing kernel. | 4.6 "Clock", M1 in-kernel SCLK and in-model probe, R10 |
| P4 | The baseline is per shape at production groups, not uniform g64; M1 must compare against production's M ≤ 16 path. | 4.6 table, section 6 replay, M1 gate, mixed-body lever |
| P5 | The probe's whole loop is 73 VALU (0.69 op/w overhead), not 62 + 0.25. | 4.2, M1 |
| P6 | The LDS reduce buffer made SK = 16 illegal and scaled with MT. | 4.3 legality, 4.5 step 1 (8-row passes) |
| P7 | 128-column WGs give 40-80 WG grids and force split tails. | 4.3 (`Wc` ∈ {32..256}), 4.5 tail, M1, R11 |
| P8 | M2's "≤ 3%" was inconsistent with A3. | 4.6 point 3, M1/M2 gates (one budget) |
| P9 | "Not modelled" was one-sided; the transform is not wider than the cast; +1.5-3% unsupported. | 4.6, 4.8 |
| P10 | The prefill estimate was optimistic. | 4.7, A3p, M8 planned |
| P11 | M8 as sketched is LDS-bound. | 4.7, Q-g7 |
| P12 | The reconstruct/crossover reasoning was wrong. | 4.7 |
| P13 | The register budget allowed illegal knob combinations. | 4.3, 4.4 (`NP·U ≤ 8`, `.vgpr_count` check) |
| P14 | Tuning-table times are not a cost model (`docs/r9700.md` rule 2). | 4.6 uses byte rooflines only |
| P15 | DFlash/MTP track plain decode in sign, not size. | 4.6, A3, Q-g8 |

**Integration review, accepted:**

| # | Finding | Where fixed |
|---|---|---|
| I1 | `EpilogueForLayout(kTrellis)` returning a new marker breaks every layer (uninitialized `pre`). | 0, 5.2, 5.3 (stays none; host entries throw on unknown values), 5.4 |
| I2 | `FallbackTuning` had no M, so prefill chunks throw. | 4.4, 5.3, `test_pick_tuning` |
| I3 | G6 would fail 3 of 5 steps (`g6_validate.ps1` hardcodes w4a16). | A4, M6 |
| I4 | `LoadShard` never parsed the trellis metadata. | 2.5, 5.1 |
| I5 | suh/svh cannot go through `ShardLoader::Part()`. | 5.1 (explicit plans, host widen, `PartBytes`) |
| I6 | M5's fused silu_mul changed numerics after A1/A2. | 4.8, 5.4 (byte-identical to v1, SHA A/B), A2 |
| I7 | No regression gate for the w4a16 path. | A6, M6, R12 |
| I8 | Tickets are never reset after an incomplete kernel. | 4.5, 5.2, R13 |
| I9 | `U` and the prescale `s` had no consistent home. | 2.3, 4.3, 4.4, 5.1, 5.3 |
| I10 | The head-layout default lived only in `Model::Load`. | 5.1 (in `Container::Load`), 5.2 |
| I11 | The M3 byte target was about 0.05 GiB low. | 2.5, A1, M3 |
| I12 | `trellis_transform.hip` needs CMake plumbing. | 4.1 |
| I13 | Mix manifests lack `stale_layers` etc. and store K as a float. | 3.3, `convert_trellis_import` |
| I14 | Part stride of the pre-transformed A; the transform grid needed K ÷ 512. | 4.8 (grid K/128), 5.3 (`part_stride`) |
| I15 | Launch counts and usage strings. | 5.1, 5.3, 5.4 |

**Narrowed (the fix is adopted, the finding's evidence is corrected):**

- **P2.** The first revision's central scenario already assumed no overlap; what changes is the
  framing (per-SIMD budget instead of lane-op rates with an ad-hoc β) and the explicit overlap/β
  measurement.
- **P4.** The review's k/v ratio (0.48-0.50) came from tuning-table times; the byte roofline gives
  0.39. The review's mixed-body lever omits that trellis excludes the q2ab rotation, so the kept
  w4a16 tensors would be unrotated; it is kept only as a post-M5 lever with its own KL run.
- **P10.** The review's "linears are about 79% of a chunk" mixes table sums with a measured chunk
  time; the measured GEMM share at T ≤ 64 is 66.8-73.2%. With both, the estimate is 0.74-0.84×,
  and M8 is still planned.
- **P15.** The 36.8 ms round with a 6.3 ms drafter (`docs/tp.md:111`) predates today's 108 tok/s
  DFlash, so its percentages are not reused; the round composition is re-measured (Q-g8).
- **I2.** Under the first revision's fallback, o/out_proj would have got SKG = 3 (legal at K = 6144)
  rather than 2; the conclusion (every prefill chunk of k, v, z, o, out_proj and down throws)
  stands. SKG is now also rounded down to a power of two.
- **I8.** A failed launch runs no WG, and neither a host exception nor a TP abort stops an enqueued
  GEMM (a TP abort only makes the spinning all-reduce kernels exit, `docs/tp.md:276`), so only a
  kernel that did not complete (a device fault or TDR) can leave a ticket non-zero. The zeroing is
  kept because it costs one `hipMemsetAsync` per `Reset()`.
- **I10.** The cited tests pass explicit w4a16 or bf16 head layouts (`test_keep_bf16.cpp:152`,
  `test_gdn_layer.cpp:76`, `test_final_lm_head.cpp:63`), so none breaks today; the risk was new
  callers passing the body layout.
- **I15.** The review's "48 fewer launches" is also off for q2ab_hv2_q3: v1 is 34 fewer (48 GDN
  Hadamards + 2 residual rotations − 16 extra transforms, because q2ab keeps attn.k/v bf16 in layers
  32-63), and M5 is 194 fewer, not 208.

---

## 10. Measured

### 10.1 M1 (2026-09-26, device 1; libr4d `67528dd`)

Full data is in `D:\models\r4dx\trellis-m1\` (start with `m1_report.md`).

**Tests.**
- `test_trellis_decode`: 2957 checks, all bit-exact.
- Whole suite (`ctest -LE tp2gpu`): 92 pass, 1 skipped for an absent golden, 0 fail.
- Every one of the 50 instantiations runs exactly 62 VALU per tile pair per k-tile. VGPRs range
  from 54 to 188, with no scratch.

**Gate: GO at M = 1 and at M = 8, provisional on the in-model clock.** The replay runs one decode
step's 336 GEMMs at full size. Trellis `_raw` runs without the output transform; w4a16 runs
q2ab_hv2_q3's production groups and tunings. S is taken per alternating pair of chains; each row
pools 3 runs.

| trellis tuning | M | trellis ms/token | w4a16 ms/token | S median [p25..p75] |
|---|---|--:|--:|---|
| fallback (4.4) | 1 | 21.10 | 22.81 | +1.73 [+1.68..+1.76] |
| fallback | 8 | 21.86 | 23.56 | +1.70 [+1.66..+1.77] |
| swept at M = 1 | 1 | 20.21 | 22.84 | +2.63 [+2.59..+2.64] |
| swept at M = 1 | 8 | 22.10 | 23.60 | +1.49 [+1.44..+1.52] |

The bar was S ≥ +0.16 ms. Trellis reaches 0.94-0.99 of w4a16's effective bandwidth, against a
byte ratio of 0.879.

**What this changes in 4.6.** The expected case there failed A3. The measurements are better:

- The decode ops issue at full rate. `v_mul_lo_u32` is quarter rate and is not on the path.
- WMMA and VALU do not overlap. But at 2 or more waves per SIMD, the ALU hides behind memory:
  β = 0.01-0.03, against 0.25 at 1 wave.
- Split tails have no measurable cost.
- The clock sag is 3-6% below w4a16's in the same run (2.78-3.15 GHz). The verdict holds unless the
  in-model SCLK falls below 2.63-2.83 GHz.
- **Prefill** at M = 64 on the MLP pair is at parity with tuned w4a16 (474 vs 479 µs), not the
  29-45% loss 4.7 expected. So M8 may be unnecessary.

**For M2.**
- M2's added work X may be up to about 1.3 ms per token (bound by M = 8 against tuned w4a16).
- Pick each shape's tuning for M = 1 and M = 8 together, timed on whole chains. The M = 1-only
  sweep gains 0.89 ms at M = 1 but loses 0.24 ms at M = 8.
- Use NT = 1 for M ≤ 16, drop SKG = 8, and keep at least 2 waves per SIMD.
- Add the in-model clock probe and the DFlash/MTP round timing early.
- AFFINE is not needed.
- The one weak shape is k/v at 1024 columns (335-406 GB/s vs 465-478).

### 10.2 M2 (2026-09-27, device 1; libr4d `67528dd` plus the uncommitted M2 tree)

Full data is in `D:\models\r4dx\trellis-m2\` (start with `m2_report.md`).

**Built.**
- libr4d: `r4d_gemm_trellis_nt_m64`, the whole linear (4.5: 8-row LDS reduction, tickets, the fp32
  FWHT of `r4d_fwht128.h`, svh, one bf16 rounding), two A parts, and KB = 5. The K loop is exactly
  62 VALU per tile pair and k-tile at KB = 4 and 75 at KB = 5 in all 100 instantiations; at most
  189 VGPRs, no scratch. `test_trellis_gemm.py` runs it through `r4d.pyd`.
- r4dx: `r4dx_trellis_input_bf16` (4.8), `test_trellis_input`, `test_trellis_gemm`, and the bench's
  `--modes full`. The tuning rows are in `tests/kernels/gemm_tuning_table_trellis.inc`; M4 moves the
  file into `src/model` when it wires it in.
- Three things the build needed:
  - at KB = 5, a `readfirstlane` on the prefetch step select, which the compiler otherwise put in a
    VGPR ("illegal VGPR to SGPR copy");
  - in the transform, an empty `asm` that keeps the fp32 product (4.8);
  - in the epilogue, barriers that order LDS only. `__syncthreads()` also waits for every global
    access and invalidates L0 at each barrier in WGP mode. With svh loaded before the reduction,
    the output side at M = 1 fell from 0.31-0.33 to 0.23 ms per token.

**Tests** (device 1). All three trellis tests pass:
- `test_trellis_input`: 360 checks, bit-exact.
- `test_trellis_decode`: 3152 checks, all 98 instantiations.
- `test_trellis_gemm`: 10935 checks.
  - Accuracy: worst element 0.405 of the tolerance; ‖C−y‖/‖y‖ ≤ 1.78e-3.
  - The epilogue is checked bit for bit against `bf16_rn((FwhtLds(S)·svh)·out_scale)` of the
    `_raw` sums, in 512 calls. The check fails on a reassociated product or a reordered FWHT.
  - Row identity holds for every table row and for the fallback.
  - 100 repeats give identical bytes at SKG = 4, at Wc = 32, and at SKG = 8.
- Whole suite (`ctest -LE tp2gpu`): 89 pass, 0 fail, and `test_kernel_bandwidth` is skipped for an
  absent golden. Only 90 tests were registered, because the reference venv had gone and the six
  `reference_*` Python tests are not registered without it.

**Gate: pass at M = 1 and M = 8, for K4 and the 4.5 bpw mix.** The replay runs one decode step's 336
linears with the table tunings, against q2ab_hv2_q3's GEMMs plus its casts, GDN Hadamards and
residual rotations. Each row pools 36 pairs; brackets are p25..p75 of the pairs.

| rate | M | trellis ms/token | q2ab ms/token | S | X | X − S |
|---|---|--:|--:|--:|---|---|
| K4 | 1 | 20.64 | 23.18 | +2.71 | +0.17 [+0.14..+0.19] | −2.53 [−2.55..−2.51] |
| K4 | 8 | 21.36 | 24.11 | +2.69 | −0.07 [−0.08..−0.04] | −2.74 [−2.77..−2.73] |
| mix | 1 | 23.16 | 23.19 | +0.20 | +0.17 [+0.12..+0.19] | −0.04 [−0.06..−0.02] |
| mix | 8 | 23.74 | 24.14 | +0.34 | −0.07 [−0.10..−0.04] | −0.41 [−0.45..−0.39] |

The bars were X ≤ 0.30 ms and X − S ≤ 0.14 ms. Both chains ran at 3187-3237 MHz.

- **Where X comes from** at M = 1: the 336 transforms (0.320 ms) cost 0.008 ms more than q2ab's
  320 casts, 48 GDN Hadamards and 2 rotations (0.350 ms). The epilogue and split tails add
  0.23 ms (K4) and 0.19 ms (mix). At M = 8 both are at or below zero.
- **Outside X.** v1 runs the plain silu_mul and attention gate-mul where q2ab runs their Hadamard
  forms.
  - silu_mul: +0.105 ms at M = 1 (one workgroup per row), 0 at M = 8.
  - gate-mul: −0.015 ms.
  - **v1 end to end** is (X − S) plus these: K4 −2.44 / −2.76 ms and mix **+0.055** / −0.43 ms per
    token at M = 1 / 8. The mix at M = 1 is inside A3's 0.14 ms margin.

**For M4.**
- **Wire the table.** Make 5.3's struct changes: `kTrellis` last, `SKG`/`U` in `LinearTuning`,
  `rate` in `GemmTuningRow`, and `BestRow` matching on `rate`. Include the file in its own
  namespace, as with the TP = 2 table. There are no TP = 2 rows and no rows for M > 16, so those
  take the fallback. The fallback is now capped at SKG = 4 (4.4).
- **Tickets and workspace.**
  - Tickets: one `tickets_bytes(N)` slice per linear, cleared by `zero_tickets`.
  - Workspace: `ws_bytes(M, N, SKG)`. The Wc = 32 rows (out_proj/o, k/v, down) take the split
    path, so they need `ws` even at SKG = 1.
- **The transform** takes host arrays of suh and out pointers (4.8, 5.3).
- **silu_mul.** Give trellis a wide-grid silu_mul, or M5's fused variant; that removes the
  +0.105 ms.
- **Not done in M2,** because the files were outside its scope:
  - the `GemmTrellisNtM64` wrapper in `r4d.hpp`;
  - the in-model clock probe;
  - the DFlash/MTP round timing.

### 10.3 M3 (2026-09-27, CPU only; converter)

Logs are in `D:\models\r4dx\trellis-m3\`. The recipe is `tools/quant2/trellis_convert.ps1`.

**Built** as section 3 specifies: `r4dx-convert --trellis-from` in `src/convert/main.cpp` and
`trellis_import.hpp`, with the `trellis` `LayoutSet` in `linear_layouts.hpp`. Changes to the plan
in 3.3:

- Steps 7-8 (file sha256, tensor dtype and shape) run after planning and before the header is
  written, so a changed oracle directory produces no output.
- **Stricter verify.** Step 10 also requires `|rel - rec| / rec <= 1e-4`, on top of the spec's
  `0.02 rec + 1e-4`. The spec's 2% cannot see one wrong 16×16 tile of a large tensor (about 6e-4 of
  rel on a gate_up) or a check that skipped part of a tensor (a mutation that drops one 128-row block
  from the sums moves rel by up to 0.56% on the test fixture). The CPU reconstruction reproduces the
  oracle's rel to 2e-8 on the real containers and 2e-7 on the fixture, so 1e-4 leaves a wide margin.
- **Where the output lives.** The container is written as `<output>.partial` and renamed to
  `<output>` only after the check passes. Any earlier `<output>` or `<output>.verify-failed` is
  removed first. A failed check renames the file to `<output>.verify-failed`. A conversion that dies
  (in the emit pass or inside the check) leaves only `.partial`. So `<output>` never holds an
  unfinished or unchecked container.
- **Durability.** The data is flushed to disk before the file is closed. The header patch flushes
  the file again before it writes, and flushes the patch after.
- **What the header records.** `r4dx_convert_run.trellis.verify` is `{mode, tolerance, result,
  worst, worst_tensor, checked, failed}`, with `worst` as a number (3.3's `verify: {mode, worst}`).
  The check patches the numbers in place of the fixed-width "pending" placeholder of `result`, as
  more members of the same object. JSON whitespace pads the space after the last value.
- **`--trellis-verify none`** exists in debug builds only, as 3.3 says. A release converter refuses
  it. A debug build records `result` "not run (--trellis-verify none, a debug build)" with no
  numbers.

**For M4 (loader).** Accept a container only when `verify.result` starts with "pass", or check
`failed == 0` and `checked` equal to the HF tensors of `quant.trellis.linears`. "pending" (the
conversion died), "FAILED" and "not run" are refused. A `.partial` or `.verify-failed` file is never
under the intended name anyway.

**Recipe.** These flags cover everything outside the body. They are q2ab_hv2_q3's flags without the
rotation.

```
--layouts w4a16 --lm-head w4a16 --no-bf16 --mtp on --vision on --kv-calib <kvcalib-full>
--quant search --imatrix <imatrix> --hessian-dir C:\AI\r4dx-hessian\hessian-v2 --ldlq .
--w4a16-group-rule "^lm_head$=32"
```

With these flags:

- lm_head is w4a16 g32. It and the MTP head's qg, o, gate_up and down (w4a16 g64) are LDLQ'd against
  hessian-v2's unrotated keys.
- The other q3 rules and the attn.k/v `--keep-bf16` are dropped: no body linear has a w4a16 layout
  any more.
- The lm_head is `w4a16` rather than `4bit`, because `--layout trellis` loads the w4a16 head.

The script pins the manifest to the one whose KL was measured: `weights_override.manifest_sha256` of
the A0 run's `D:\models\r4dx\kl-trellis\<oracle>\reference_run.json` (K4m `7e9037f4…`, mix4.5m
`48a2eacb…`), so a manifest changed since then is refused. `-ManifestSha256` overrides the pin. The
log records the converter's path, time and sha256.

611 tensors outside the body are byte-identical to q2ab_hv2_q3's, in both the K4m and the mix
container: lm_head.w4a16, mtp.* (except its two layernorms), vision, the embeddings, final_norm, the
descales, q/k norms, conv1d, A_log, dt_bias and the GDN norm. This is `compare_containers.py` with
every body linear, the rotation, the layernorms, gdn.in_proj_a/b and lm_head's dropped mxfp4/w4a8
copies excluded. The recipe's own `-Compare` set is 356 of them.

**Tests.** `convert_trellis_import` covers the pair grid, the codebook, the ring tables and the
LayoutSet. On the fixture that `tools/reference/trellis_import_golden.py` writes:

- the decode of 26 oracle tensors matches `decode_words` bit for bit;
- the regrid matches `to_pair_grid` byte for byte, in both manifest forms;
- the converter runs in both forms, and its metadata is checked, including `verify.worst <= 1e-4`;
- the non-body tensors equal those of a conversion without `--trellis-from`;
- the `--lm-head bf16` twin, `--keep-bf16` and `--trellis-prescale-log2` work;
- every refusal in 3.3 is hit, plus one file recorded with two sha256, a prescale outside
  [-16, 16] and (release builds) `--trellis-verify none`;
- a rec off by 2× and one off by 1e-3 (inside the spec's 2%) both fail, with the numbers in the
  header, the file renamed to `.verify-failed`, and neither `<output>` nor `.partial` left.

The fixture checkpoint's error is deliberately uneven: `W_hat·(1 + a·g)`, with the amplitude a drawn
per 16×16 tile from [0.002, 0.02]. With an even error (plain bf16 rounding), a check that covers only
part of a tensor reproduces rel exactly and cannot be caught.

The test keeps `tiny_k4.r4dx` and `tiny_mix.r4dx` in the build tree's `tests/convert/trellis_import/`
for M4; a test that loads them declares `FIXTURES_REQUIRED trellis_tiny`. Their config is the real
model's in miniature: hidden 256, 4 heads of 256, 2 KV heads, GDN 2 key and 8 value heads of 128,
intermediate 1024, vocab 128. The test checks it against the runtime's own code:

- `ModelConfig::FromJson` parses it;
- every trellis linear has the [N, K] that `Container::Load` asks for;
- `ModelConfig::Shard` accepts it at TP = 2, and every trellis rank range is a multiple of 128.

All 15 `convert_*` tests pass, including `convert_rms_hessian`'s pinned digest of an unrotated
container built without the flag.

| container | oracle | time | file | decode bytes | verify, worst \|rel - rec\|/rec |
|---|---|--:|--:|--:|---|
| `qwen38-27b-trellis-k4m.r4dx` | K4m | 1.7 min | 15.66 GiB | **12.130 GiB** | 400/400, 1.79e-8 |
| `qwen38-27b-trellis-k4m-lmbf16.r4dx` | K4m, `--lm-head bf16` | 1.6 min | 17.29 GiB | 13.758 GiB | 400/400, 1.79e-8 |
| `qwen38-27b-trellis-mix45m.r4dx` | mix4.5m | 1.8 min | 17.08 GiB | **13.546 GiB** | 400/400, 1.96e-8 |

Decode bytes are measured by `tools/quant2/decode_bytes.py`, which gives 13.680 GiB for q2ab_hv2_q3.
The twin's 1843 non-lm_head tensors are byte-identical to K4m's. The reconstruction check takes 22-39
s. Hashing the oracle files takes about 6 s. An independent numpy check finds the pair grid, suh and
svh byte-equal to the oracle files for 15 linears across both containers. They include gate_up at
KB = 4 and 5, down with K = 17408, and attn.k/v with N = 1024.
