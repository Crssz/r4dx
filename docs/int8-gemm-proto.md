# int8 x int8 prefill GEMM: a go / no-go prototype, before anyone builds the production kernel

Status (2026-10-07, branch `int8gemm`, off `int8q`): the bench is written, builds, its CPU checks pass and the
ISA is inspected. **No GPU run has happened yet** (the session that wrote it was CPU-only by rule); every result
table below is EMPTY and the section "Predictions" is a cycle-budget model, not a measurement. The commands that
fill it are in "How to run".

The question (docs/int8-prefill.md ends on it): accuracy is not the obstacle to an int8 prefill GEMM (full w8a8,
one scale per 128 K on both sides, costs +0.0006 mean KL), so can an int8 x int8 GEMM at M = 256 beat the shipped
f16 trellis kernel (docs/trellis-m256.md) by enough to be worth building? The earlier fp8 W8A8 dense prototype
(2026-10-05) reached 134.6 TF/s at M = 256 against a 284 TF/s fp8 WMMA probe peak, and the shipped M = 64 kernel
ran at about 44% of the f16 WMMA peak, so "2x the WMMA rate" is not a speedup by itself. This prototype measures
the speedup instead of assuming it.

**Decision rule** (set before any number existed):

| outcome | action |
|---|---|
| dense int8 GEMM (item 2, no decode, no rotation epilogue) < 1.5x the shipped f16 trellis kernel | the idea is dead, whatever the decode costs |
| trellis int8 (item 3) >= 1.4x shipped on the MLP pair (gate_up + down) at M = 256 | GO: build the production kernel |
| trellis int8 < 1.15x shipped on the MLP pair | STOP |
| in between | inconclusive: the "speed bounds" rows below say which accuracy-for-speed trade (coarser scales) would clear 1.4x; that trade needs its own KL gate |

## What was built (all under `tests/kernels/`, bench-only)

| file | what |
|---|---|
| `int8_gemm_proto_ref.h` | the layouts, the quantizer definitions and the CPU references (header-only, hipcc and plain C++) |
| `int8_gemm_proto_kernels.h` | the device code: `i8g_kernel` (dense and trellis int8 GEMM), the activation producer `i8g_quant_act`, the scale-table builder `i8g_wscale`, the int8 matrix dump `i8g_dump_w`, the exact reference `i8g_ref_cols`, the host launch / legality |
| `tool_int8_gemm_proto.hip` | the bench: `probe`, `selftest`, `verify`, `time` |
| `test_int8_gemm_proto_cpu.cpp` | ctest `test_int8_gemm_proto_cpu`, pure CPU, 0.3 s |
| `build_int8_gemm_proto.ps1` | hipcc build (no CMake target, like `build_m256_bench.ps1`), `-Isa` prints every kernel's VGPRs, scratch and instruction counts |

No libr4d unit, no `src/` file and no CMake target other than the CPU test was touched: the shipped kernels and
their ISA are as on `int8q`. The bench links the shipped `r4d_gemm_trellis_nt_m256` unit for its baseline.

### The kernel

`i8g_kernel<TRELLIS, KB, FWHT, RESC, SKW>` has the structure of `r4d_gemm_trellis_nt_m256`: a workgroup is
4 x SKW waves; wave (rg, ks) owns rows 64 rg .. 64 rg + 63 and K slice ks, 4 row tiles x 2 accumulator fragments
(a 32-column tile pair), and one (tile pair, k-tile) block of weights is decoded once per workgroup and shared
through a double-buffered LDS slot. What differs:

* **WMMA**: `v_wmma_i32_16x16x16_iu8`, A and B signed, int32 accumulate. A is stored in a fragment-ready layout
  (`A8`, one 8 B load per lane per (row tile, k-tile), the 4 row tiles of a row group 1 KiB contiguous), so the
  producer (a fused transform in production; `i8g_quant_act` here, timed separately) writes the permuted bytes.
  The k order inside a fragment is the trellis decode's own (k = 8 (e >> 2) + 4 h + (e & 3)), so the decoded B
  fragment needs no permutation. A dot product does not care about the order of its terms, so the hardware's own
  iu8 k order never enters: only that A and B agree, which they do by construction.
* **A step is a 128-K block** (8 k-tiles), the scale group. Each wave decodes TWO of the block's eight blocks
  (k-tiles rg and rg + 4) into the LDS slot, one barrier, then every wave reads the 8 blocks' fragments (8 x
  `ds_load_b128`) and runs **4 passes**, one row tile each: 8 k-tiles x 2 fragments of WMMA into two int32
  accumulators (16 VGPRs), then `acc_f32 += float(int32) * (sa[row] * sw[col])` into the fp32 accumulators
  (cvt + mul + fma per element). Holding only one pass's int32 accumulators keeps the kernel at <= 192 VGPRs
  (8 waves per SIMD, two 16-wave workgroups per WGP), where keeping all 8 tiles' int32 accumulators next to the
  fp32 ones would not.
* **Dense (TRELLIS = false)**: the weights are int8 already (`W8`: the trellis GEMM's block order, 16 B per lane,
  1 byte per weight) and the "decode" is one 16 B load. This is the ceiling: no decode, and with FWHT = false no
  rotation epilogue either.
* **Trellis (TRELLIS = true)**: the weights are the shipped kernel's words and the decode is
  `r4d_trellis_k4_decode` / `k5_decode` unchanged. The two f16 fragments are then quantized with the
  precomputed table: `v_pk_fma_f16(w, rs, 1536)` rounds `w * rs` to an integer INSIDE an f16 (the f16 spacing
  from 1024 to 2048 is exactly 1, so the single fma rounding is `rint`, ties to even), whose low byte is the
  int8 (512 is a multiple of 256); one `v_perm_b32` packs four. 16 `v_pk_fma_f16` + 8 `v_perm_b32` per kb per
  wave, no cvt, no clamp (the group's amax maps to 127 and f16 `rs` is within 2^-11 of its value, so
  |w rs| <= 127.07). CPU-checked bit for bit against `rint(w * rs)` over every f16 `w` and 330 `rs`, ties
  included.
* **Scale tables**: `SW[kb][N]` fp32, one per (column of Q, 128 k), K/128 x N x 4 B (3% of the int8 weights,
  the same size `R4DX_FAKEQ_W=col128` costs). Its value is `s_eff = 1 / rs` with `rs = f16(min(127 / amax,
  60000))`, so dequantization (q * s_eff) is the inverse of exactly the grid the quantizer rounded on (the
  kernel recomputes `rs` from the table with the hardware reciprocal; CPU-checked idempotent, even with the
  reciprocal off by 4 ulps). `SA[kb][256]` is the activation scale (`s = amax / 127`, `q = clamp(rint(x / s))`,
  `R4DX_FAKEQ_ACT=blk128`'s rule, IEEE division).
* **Epilogue** (FWHT = true): the shipped kernel's, copied: LDS reduction over the SKW slices, fp32 partials in
  `ws`, ticket, the last block sums the SKG partials, FWHT-128, svh, out_scale, bf16. The slicing is free here
  (there is no bit identity with the shipped summation order to keep): SK = SKW, SKG in {1, 2, 4, 8}, whole
  128-K blocks per slice (mlp.down's K = 17408 = 136 x 128 takes SKW x SKG in {2, 4, 8}). FWHT = false is a plain
  reduce-and-convert.
* **Speed bounds** (RESC = 1, 2, 3; not the per-128 math, each verified against its own exact reference):
  1 = both scales per row / column only, one rescale at the end (the int32 accumulators run over the whole
  slice, no per-128 VALU at all); 2 = activation scale per row, weight scale per 128 (cvt + fma per element);
  3 = the reverse. They exist because the per-128 rescale is the largest VALU cost (see "Predictions") and
  these rows price what a coarser scale would buy.

### What the CPU session verified

* `test_int8_gemm_proto_cpu` (26 checks, pure CPU): the fragment maps and the A8 / W8 layouts are bijections and
  agree with the decode's lane map; the f16 software conversions; the quantizer trick, its ties, the table's
  idempotence; the quantizer against the fp32 rule of `R4DX_FAKEQ_W` (0.58% of weights differ by one LSB on
  Gaussian groups, the error RMS is identical to 3 digits: 0.648% of the weight RMS); a software-WMMA emulation
  of the kernel's whole fragment chain INCLUDING its K slicing (the A8 / W8 loads, the 16 x 16 x 16 tile, the
  accumulator row / column map, the scale indexing, every RESC mode, five (SKW, SKG) splits) against the
  exact-integer fp64 reference, max difference 1e-5 of the output RMS; trellis words -> `trellis_ref` decode ->
  table -> int8 matrix for KB 4 and 5. The w8a8 product of Gaussian data differs from the f16 product by 0.92%
  relative RMS.
* the ISA (`build_int8_gemm_proto.ps1 -Isa`, ROCm 10.1.0, gfx1201): every `i8g_kernel` instantiation (36)
  compiles to 145-192 VGPRs, **0 bytes of scratch**, 64 `v_wmma_i32_16x16x16_iu8` in the K loop body, 11 workgroup
  barriers in the whole kernel (1 per kb in the loop). The trellis KB 4 kernel's phase D (per wave, per kb: two blocks)
  is 32 `v_mad_u32_u16` + 32 `v_pk_mad_u16` + 32 `v_sad_*` + 12 `v_alignbit_b32` (the decode, 124 VALU) + 16
  `v_pk_fma_f16` + 8 `v_perm_b32` + 2 `v_rcp_f32` + 2 `v_cvt_f16_f32` (the quantizer). A pass is 16 WMMA, then
  16 `v_cvt_f32_i32`, 16 `v_mul_f32`, 16 `v_fmac_f32` (about 23 of the 32 mul / fmac in `v_dual_*` pairs).
  The probe kernels' loops are 8 independent WMMA chains and nothing else.

What the CPU session could NOT check: that the device code does what the emulation says (the selftest below
does, on the GPU, against CPU references: it is the first thing to run), and every number in a time column.

## How to run (device 1, machine otherwise idle; outputs under `E:\models\r4dx\int8gemm\`)

The exe is built (CPU only) at `E:\models\r4dx\int8gemm\obj\tool_int8_gemm_proto.exe`. To rebuild:
`powershell -NoProfile -File C:\Users\pay20\dev\r4dx-int8g\tests\kernels\build_int8_gemm_proto.ps1` (about a
minute; `-Lite` instantiates only skw 4, `-Isa` prints the resource table).

```
# 1. the WMMA probe: peak f16 / bf16 / iu8 / fp8 TOPS and the shader clock (a few seconds)
powershell -NoProfile -Command "`$env:HIP_VISIBLE_DEVICES='1'; & E:\models\r4dx\int8gemm\obj\tool_int8_gemm_proto.exe --mode probe --out E:\models\r4dx\int8gemm\probe.json 2>&1 | Tee-Object -FilePath E:\models\r4dx\int8gemm\probe.log; exit `$LASTEXITCODE"
# 2. the selftest: every kernel against CPU references on small shapes; MUST pass before any timing means anything (seconds)
powershell -NoProfile -Command "`$env:HIP_VISIBLE_DEVICES='1'; & E:\models\r4dx\int8gemm\obj\tool_int8_gemm_proto.exe --mode selftest 2>&1 | Tee-Object -FilePath E:\models\r4dx\int8gemm\selftest.log; exit `$LASTEXITCODE"
# 3. smoke: one class, quick protocol (about a minute)
powershell -NoProfile -Command "`$env:HIP_VISIBLE_DEVICES='1'; & E:\models\r4dx\int8gemm\obj\tool_int8_gemm_proto.exe --mode time --shape gate_up --kb 4 --quick 2>&1 | Tee-Object -FilePath E:\models\r4dx\int8gemm\smoke.log; exit `$LASTEXITCODE"
# 4. everything: probe + selftest + verify + time, all seven classes, KB 4 and 5 (guess: 10 to 20 minutes)
powershell -NoProfile -Command "`$env:HIP_VISIBLE_DEVICES='1'; & E:\models\r4dx\int8gemm\obj\tool_int8_gemm_proto.exe --mode all --out E:\models\r4dx\int8gemm\all.json 2>&1 | Tee-Object -FilePath E:\models\r4dx\int8gemm\all.log; exit `$LASTEXITCODE"
```

Options: `--shape gate_up|down|qkv|z|out|qg|kv|all`, `--kb 4|5|both`, `--rounds 11 --batch 6` (the m256 bench protocol:
`batch` cold copies of every weight tensor, every round times every job once, medians of per-round values and of
per-round ratios), `--noverify`, `--no-resc` (skip the speed-bound variants), `--verbose` (every job's row),
`--simds 128` (for the clk-per-WMMA print). Run the probe first and alone: its TOPS and clock are the yardstick.

What the bench prints per (class, KB): the shipped f16 plan (the configuration `PlanTrellisM256` picks, rebuilt
from the tuning table and the host check) and the best of the other legal ones; the best dense int8 kernel with
the trellis epilogue and without; the best trellis int8 kernel; the speed-bound variants; the activation
quantizer; time, TF/s (2 M N K / t), and the speedup against the shipped plan. A job marked `!` failed its
verification and is excluded from every summary. Then the MLP pair and all-class (layer-weighted) summaries and
the decision rule.

**Verification inside the bench** (so a number is never reported for a wrong kernel): `selftest` runs the single
iu8 WMMA against a software tile, the activation quantizer, scale table and int8 matrix against the CPU decode
(`trellis_ref.hpp`) bit for bit, and every legal configuration of every kernel on K = 1024, N = 256 against the
exact fp64 reference (fine and coarse scale masks, FWHT or plain, SKG 1..8); `verify` runs every timed
configuration on the real shapes on five whole 128-column groups (first, last, three pseudo-random; all 256
rows) against the GPU's exact-integer reference (`i8g_ref_cols`, int32 sums per 128 K, fp64 scaling), requires
the trellis kernel to equal the dense kernel on the same int8 weights BYTE FOR BYTE (same summation order, same
weights), and prints the relative RMS difference to the shipped f16 trellis output (the fake-quant level:
about 0.5 to 2%).

## Predictions (a cycle-budget model, to be compared with the measurement, NOT a result)

Model: WMMA and VALU do not overlap on a SIMD (docs/trellis-kernel.md 4.7), so a wave's K-loop time per 128 K is
`WMMA + VALU + small`. The probe will give the iu8 WMMA clock count; assume 8 clk (2x f16's 16) here. Per wave per
128 K (64 WMMA):

| kernel | WMMA | decode + quantizer | rescale | other | total | vs shipped loop |
|---|---|---|---|---|---|---|
| shipped f16 trellis, KB 4 | 64 x 16 = 1024 | 124 | none | ~20 | ~1170 | 1.00 |
| dense int8 (ceiling) | 64 x 8 = 512 | 0 | ~165 | ~20 | ~700 | 1.67 |
| trellis int8, KB 4 | 512 | ~165 | ~165 | ~20 | ~860 | 1.36 |
| trellis int8, weight or activation scale coarse | 512 | ~165 | ~110 | ~20 | ~810 | 1.45 |
| trellis int8, both scales coarse | 512 | ~165 | ~0 | ~20 | ~700 | 1.67 |

(rescale: 4 passes x about 40 issue slots; "other" is addressing, waits and the barrier.) The shipped kernel's
epilogue (about 9% of gate_up KB 4) is common to the trellis rows, which dilutes the loop ratios by about 5% of
themselves. So the model puts the dense ceiling around 1.5x (right on its own gate) and the trellis kernel at
about 1.3x: in the inconclusive band, which is why the speed-bound rows exist. The reason is structural: at
128-K scale groups the rescale costs 3 VALU ops per output element per 128 K, 0.09 to 0.075 clk of VALU against
0.125 clk of int8 WMMA per element, i.e. 60 to 75% of the WMMA time, and the decode that the shipped kernel
already pays comes on top. What would move it: coarser groups on one side (the rescale scales with 1 / group
size), fewer ops per element, or a higher iu8 / f16 ratio than 2.

How to read the probe against that: `iu8 / f16 TOPS` below about 1.6 makes the model's WMMA row worse than 512
and the dense ceiling fails its gate on its own; the fp8 row is the harness check against the known 284 TF/s.

## Results (to fill from the GPU run; machine idle, device 1, ROCm 10.1.0)

Probe (`probe.log`):

| WMMA | TOPS | shader clock (MHz) | clk per WMMA per SIMD |
|---|---|---|---|
| f16 | | | |
| bf16 | | | |
| iu8 | | | |
| fp8 | (284 on 2026-10-05) | | |

iu8 / f16 = ____ ; selftest: ____ (all pass / failing checks); verify: ____ (configurations passing; trellis ==
dense byte for byte; relative RMS difference to the shipped f16 output ____).

Per class, KB 4 (time in us, median; speedup against the shipped plan; `best` = best configuration of the sweep):

| class (layers) | f16 shipped | f16 best | dense + epilogue | dense plain | trellis int8 | trellis TF/s | trellis x | dense plain x | config (skw skg) |
|---|---|---|---|---|---|---|---|---|---|
| mlp.gate_up (64) | | | | | | | | | |
| mlp.down (64) | | | | | | | | | |
| gdn.in_proj_qkv (48) | | | | | | | | | |
| gdn.in_proj_z (48) | | | | | | | | | |
| gdn.out_proj, attn.o (64) | | | | | | | | | |
| attn.qg (16) | | | | | | | | | |
| attn.k, attn.v (32) | | | | | | | | | |
| **MLP pair** | | | | | | | | | |
| all classes, layer-weighted | | | | | | | | | |

KB 5: the same table.

A quantizer (the activation producer run as a stand-alone kernel; fused into the existing input transform in
production): gate_up input (K = 5120) ____ us, down input (K = 17408) ____ us.

Speed bounds, MLP pair (coarser scales; each needs its own accuracy gate before it counts): trellis int8 with
both scales coarse x____, activation per row only x____, weight per column only x____ (dense, both coarse
x____), KB 4 and KB 5.

**Decision**: ____ (rule above; MLP pair, KB 4: x____, KB 5: x____; dense ceiling x____).

## If the answer is GO (what the production kernel still owes)

* The A quantizer fused into the producers (the shared `r4dx_trellis_input_bf16` / silu_mul / gate-mul
  producers write `PreQuantizedActivation`; they would write A8 + SA instead of f16): its unfused time above is
  an upper bound on what it adds.
* The scale table: f32 here; f16 / bf16 storage halves it (0.8 GB -> 0.4 GB for the model) and `s_eff = 1 / rs`
  must stay consistent with the quantizer (store `rs` as f16 directly and derive nothing).
* The accuracy gate for THIS quantizer: `R4DX_FAKEQ_W` measured the fp32 rule (`rint(w / s)` with clamp, fp32
  `s`); this kernel's is the f16-`rs` rule above. 0.58% of weights differ by 1 LSB, the error RMS is the same to
  3 digits, but the KL harness (`kl_rung4.ps1`, w3 in docs/int8-prefill.md) should run on a variant that uses
  exactly it before anything ships.
* The shapes the bench does not cover: TP = 2 shards, the 64-row tail chunks (they stay on the f16 kernels), the
  two-part `n_split` linear (gate / up with two activation matrices: A8 / SA twice), `attn.k / v`'s small N.
* Real weights and activations: the bench decodes random trellis words (Q is about Gaussian, like the model's)
  and uses Gaussian activations; the kernels' speed does not depend on the values, the clock under WMMA load
  can.

## Not done / caveats

* Nothing was run on a GPU. The kernels were written, compiled, inspected at the ISA level and checked against a
  CPU emulation of their fragment chain; a device bug the emulation cannot see (an LDS or barrier slip, a
  builtin's behaviour) would show as a selftest failure, not as a wrong timing.
* One kernel family with a sweep of (SKW, SKG) only. Not tried: NP = 2 (64-column tiles), a fragment-direct
  dense kernel without the LDS stage, software-pipelining a pass's rescale behind the next pass's WMMA (it adds
  24 VGPRs, and VALU and WMMA do not overlap on a SIMD anyway), M = 512 (the chunk is 256). The model says the
  decode and the rescale VALU, not the WMMA, set the speed, so those would be worth trying only after a first
  measurement says which of them dominates.
* The probe's "clk per WMMA per SIMD" assumes 128 SIMDs (64 CUs x 2); `--simds` overrides.
