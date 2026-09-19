# libr4d kernel lessons + r4dx usage audit (gfx1201 / RDNA4, R9700)

Scope: read every kernel/header in `third_party/libr4d` and every kernel/launch site in
`src/kernels` and `src/model`, local-only, no GPU access this pass. Citations are `file:line`
against this checkout. Confidence is marked inline: **measured** = the libr4d author states a
number from hardware; **documented** = stated in a comment/doc without a number attached;
**inferred** = my own read of the code, not stated verbatim anywhere.

## 1. Micro-architectural lessons extracted from libr4d

### 1.1 WMMA fragment layouts

The one fact everything else in the library is built on, stated in the README and re-derived at
the top of `r4d_gdn_wmma.h`:

> "The wave32 WMMA fragment layout is `idx = lane % 16`, `k = 8*(e>>2) + 4*(lane>>4) + (e&3)`.
> Both A and B want K-contiguous rows, so swapping the operands of a WMMA transposes the result
> for free." (`third_party/libr4d/README.md:134-135`, re-stated at
> `third_party/libr4d/r4d_gdn_wmma.h:1-6`)

Consequences this drives everywhere:
- A `[N][M]`-shaped output store is one contiguous 8-element write; `[M][N]` costs eight scalar
  stores (`r4d_gdn_wmma.h:5-6`).
- The GDN chunk-scan holds its recurrent state **transposed**, `C[m=k-dim][n=v]`, purely because
  WMMA A/B share this lane layout and "swapping the operands of the state update is free... and it
  makes k-dim the per-lane contiguous axis, so the per-chunk writeback to `s.S` is four packed
  8-short stores instead of 32 scalar `ds_store_b16`" (`r4d_gdn_chunk_scan_k128_v128_c64_bf16.hip:271-274`).
- All GEMM/attention/GDN weight and activation packers pre-permute their operand into exactly this
  fragment order **offline**, so the runtime kernel does zero cross-lane shuffling on the weight
  path — `r4d_gemm_w4a16_nt_m64.hip:45-54`, `r4d_gemm_w4a8_nt_m64.hip:22-34,47-60`,
  `r4d_gemm_mxfp4a8_nt_m64.hip:48-59` (the mxfp4 file measured the checkpoint-order vs.
  fragment-order weight read: **1.15x at M=8, ~1.3x at M=64, 1.44x over reading checkpoint order
  directly**, table at `r4d_gemm_mxfp4a8_nt_m64.hip:62-77`).

Builtins used, by dtype:
| WMMA builtin | shape | used in |
|---|---|---|
| `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12` | 16x16x16 bf16xbf16→f32 | `r4d_dt16.h:31-33` (DT16<0>), `r4d_gdn_wmma.h:126-128` (`mma()`, used throughout GDN) |
| `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` | 16x16x16 f16xf16→f32 | `r4d_dt16.h:52-54` (DT16<1>, attention), `r4d_gemm_w4a16_nt_m64.hip:153-155` (`mmah`) |
| `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12` | 16x16x16 fp8xfp8→f32 | `r4d_gemm_mxfp4a8_nt_m64.hip:196` |
| `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12` | 16x16x16 int8xint8→int32 | `r4d_gemm_w4a8_nt_m64.hip:122-124` (`mmai`) |

**Correction to the task brief**: the brief describes "int8 iu8 16x16x32 for w4a8". The actual
builtin r4dx uses is `wmma_i32_16x16x16_iu8`, i.e. K=16 not K=32. The K=32 int8 shape does not
appear anywhere in libr4d. A `v_wmma_i32_16x16x32_iu4` (int4xint4) instruction is *named* in a
comment as a measured issue-rate reference point (**measured**, 830 TF/s,
`r4d_gemm_w4a16_nt_m64.hip:22`, repeated `r4d_gemm_w4a8_nt_m64.hip:14`) but no kernel in the
library actually issues it — every 4-bit weight is dequantized to f16, iu8, or fp8 before any WMMA
runs. That int4x-int4 instruction is unused headroom (see §4, prefill kernel design).

Matrix-instruction issue rates, all **measured** by the author and repeated verbatim at
`r4d_gemm_w4a16_nt_m64.hip:20-27` and `r4d_gemm_w4a8_nt_m64.hip:9-14` /
`r4d_gemm_mxfp4a8_nt_m64.hip:10-15`:

| instruction | TF/s | cyc/instr |
|---|---|---|
| `v_wmma_f32_16x16x16_f16` | 207 | 5.95 |
| `v_wmma_f32_16x16x16_fp8_fp8` | 412 | 2.99 |
| `v_wmma_i32_16x16x16_iu8` | 407 | 3.03 |
| `v_wmma_i32_16x16x32_iu4` (named, unused by any kernel) | 830 | 2.97 |

This table is the entire reason `r4d_gemm_w4a8_nt_m64` and `r4d_gemm_mxfp4a8_nt_m64` exist beside
`r4d_gemm_w4a16_nt_m64`: at M=64 the f16 kernel is compute-bound on the *slowest* matrix
instruction on the part (72-78 TF/s measured, `r4d_gemm_w4a8_nt_m64.hip:6-16`), and switching the
widened dtype to 8-bit is worth ~1.5-1.9x for free.

### 1.2 bf16 packing (no native convert instruction)

gfx1201 has **no** `v_cvt_pk_bf16_f32` (**measured/verified absent**, stated three times:
`r4d_common.h:25-26`, `r4d_dt16.h:5`, `r4d_gdn_wmma.h:13-14`). Every f32→bf16 conversion is
therefore software RTNE, and the whole library converges on the same trick: **two adds plus one
`v_perm_b32`**.

- Scalar RTNE: `(u + 0x7fffu + ((u>>16)&1u)) >> 16` (`r4d_common.h:27-30`,
  `r4d_dflash_conv_body.h:40-43`).
- Packed-pair RTNE in 3 instructions: `f2bf2()` = two adds (`+0x8000`, biased RTNE) then one
  `__builtin_amdgcn_perm(ub, ua, 0x07060302u)` to gather the two high halves
  (`r4d_gdn_wmma.h:16-24`).
- A **truncating** variant exists for the hot inner path: `f2bf2t()` is one `v_perm_b32` with no
  rounding add at all, "1 ULP of bf16 instead of 0.5" (`r4d_gdn_wmma.h:29-35`). The chunk-scan
  kernel's `BFR` flag chooses per-buffer which packs may truncate (intermediates W/sA/V' vs. the
  carried state/output) and the author **measured and rejected** always-truncate: it is
  1.5-2.2% faster but moves the fp64-oracle error from 3.30e-03 to 9.02e-03/1.17e-02 because those
  intermediates feed the recurrence (`r4d_gdn_chunk_scan_k128_v128_c64_bf16.hip:186-194`).
- f16 is *cheaper* than bf16 on this part for the opposite reason: `v_cvt_pkrtz_f16_f32` does two
  f32→f16 conversions in one instruction, so wherever f16 is numerically sufficient (both attention
  operands, fp8-widened values) the library prefers it — "24 VALU per 8-element P fragment against
  4" is the quoted cost delta (`r4d_dt16.h:1-8`).
- The 4-bit-weight GEMM's dequant is bf16-hostile for a different reason: gfx1201 has **no packed
  bf16 arithmetic** (`v_pk_add_bf16` does not exist) while `v_pk_add_f16` does, so the zero-point
  subtract is 4 instructions in f16 vs. 16 in bf16 (through f32 and back) —
  `r4d_gemm_w4a16_nt_m64.hip:10-18`.

### 1.3 LDS budgets, CU-mode, occupancy

| kernel | LDS layout | total LDS | occupancy note |
|---|---|---|---|
| `r4d_gdn_chunk_scan_k128_v128_c64_bf16` | `Smem{ k[64][132], A[64][68], S[64][132], W[64][68], D[64][68], ge/gv/rs/bt[64] }` (bf16 2B rows + 4x fp32[64]) | ≈60.9 KB (**inferred** from `Smem` field sizes, `r4d_gdn_chunk_scan_k128_v128_c64_bf16.hip:109-118`) | `NW` (waves/block) is 8 or 16; at NW=8 the kernel "sits on 242 VGPRs and keeping one more address live spills 146 of them" — one state tile per wave (NW=16) halves live accumulators (`:66-73`). Compiled with `-mcumode` in isolation (README.md:113 — the only TU singled out for CU mode), presumably to keep this LDS footprint private to one CU rather than shared WGP-wide. |
| `r4d_gdn_kkt_solve_k128_c64_bf16` | `KkSmem{ A[64][68], TMP[32][36], gb[64], bt[64] }` | **measured, stated explicitly**: "11 KB, where the first version wanted 31" (`r4d_gdn_kkt_solve_k128_c64_bf16.hip:90`) | `OCC` (min waves/SIMD) explicitly left at 0 (natural occupancy) after measuring that forcing 10 waves (`__launch_bounds__(NTHR,10)`) spills 148 B/lane and is **1.57x slower** (39.9us vs 25.4us) than the 8-waves-from-190-VGPRs the compiler picks unforced (`:106-114`). |
| `r4d_attn_prefill_h256_gqa6` (TILE=48, HEAD_DIM=256) | `sK[TILE*KSTR] + sV16[HEAD_DIM*VSTR]`, KSTR=264/VSTR=56 (bf16, CONTIG on) | ≈54 KB (**inferred**: 48*264*2 + 256*56*2 = 25344+28672) | Geometry is capped by LDS, stated directly: "bf16 KV at TILE 64 would need 68096 B > 65536, so the geometry stops at 48" (`r4d_attn_paged_h256_gqa6.hip:47-48`). `NWARPS=24` → 768 threads/block. |
| `r4d_attn_decode_h256_gqa6` (TILE=16) | same shape, much smaller TILE | small (**inferred**, not stated) | `NWARPS` picked per launch from `rows=q_len*gqa` (1-4), i.e. occupancy tracks the speculative-verify width rather than being fixed. |
| `r4d_gemm_{bf16,w4a16,w4a8,mxfp4a8}_nt_m64` | reduction-only scratch, `WV*(NPW)*SK*256*sizeof(float)` | caller-chosen, hard-capped and checked: `if (smem > 64*1024) throw` (e.g. `r4d_gemm_w4a16_nt_m64.hip:302`) | `__launch_bounds__(1024)`, no occupancy hint — up to 32 waves/block, WV*SK sized per shape by the offline tuning sweep (`docs/perf.md` §"GEMM tuning sweep"). No LDS staging of A or W at all; only used for the split-K reduction. |
| `r4d_gemm_bf16_nt_m16` | `WV*SK*MT` floats | tiny | No `__launch_bounds__` cap stated in the kernel itself (default). |
| `r4d_gdn_conv_prep_w4_h128_bf16` / `conv_update` | no LDS buffer at all — conv state lives in registers (`win[CP_W][CP_DPL]`) | 0 | `__launch_bounds__(HPB*32)`, HPB up to 8, narrowed at launch time if the grid is too thin to fill the device (`r4d_gdn_conv_w4_h128_bf16.hip:425-431`). |
| `r4d_gdn_recurrent_update_k128_v128_bf16_fp32state` | `wsum[THR/32]` only (fused-norm reduction) | trivial | `VS` (1/2/4) splits a head's 128 v-rows across up to 4 workgroups specifically because "the kernel is not DRAM bound there (its whole working set fits the 64 MB last level)" — occupancy is filled by splitting work, not by tuning waves/SIMD (`:43-50`). |
| `r4d_gdn_gated_rmsnorm_h128_bf16` | none (register-only, one wave per row) | 0 | `__launch_bounds__(GN_NW*32)`, GN_NW=8. |

### 1.4 Split-K and NT (non-temporal) loads

Every `r4d_gemm_*_nt_m64` kernel decomposes a block into `WV` column-tiles x `SK` k-splits of
waves, reduced through the LDS scratch **in fixed wave order**, "so the result does not depend on
how the waves were scheduled" (stated identically in all four GEMMs, e.g.
`r4d_gemm_bf16_nt_m64.hip:122-126`). This determinism claim is directly falsified by
`docs/perf.md`'s "Generated-text regression" section: retuning `SK` for prefill changed the
quantized layouts' generated text a few dozen tokens in, because a different `SK` is a genuinely
different summation order through the reduction — the kernel is deterministic *for one fixed SK*,
not across SK choices (**measured on hardware**, `docs/perf.md:69-76`).

NT loads (`__builtin_nontemporal_load`) are applied to the **weight** tile only in w4a16/w4a8,
gated by a template bool `NT`, with a **measured, non-monotonic** win: "+3 to +4% at M=8 and M=16
... and a LOSS from M=24 up on the shapes with the longest K" (`r4d_gemm_w4a16_nt_m64.hip:77-84`).
This is per-call, carried in the tuning table (`src/model/gemm_tuning_table.inc`'s `NT` field), not
a fixed compile-time choice — though `docs/perf.md`'s GEMM sweep notes `NT` was **held fixed at 1**
throughout r4dx's own tuning pass rather than actually swept (`docs/perf.md:150-153`), so the
per-M-band NT decision libr4d's own comment recommends is not yet exercised in r4dx's table.

### 1.5 `global_load_tr` (transpose load)

`load_tr_b128()` wraps `__builtin_amdgcn_global_load_tr_b128_v8i16`, "8x8 transpose of 16-bit
elements across each group of 8 lanes... the 8 addresses may be arbitrarily strided"
(`r4d_common.h:47-51`). Used exclusively in the attention kernels' `fetchV` lambdas (both prefill
and decode) to fetch V directly transposed from the paged cache in one instruction, replacing what
would otherwise be an LDS-mediated transpose — `r4d_attn_prefill_h256_gqa6.hip:204-222`,
`r4d_attn_decode_h256_gqa6.hip:135-153`. Not used anywhere in the GEMM or GDN kernels (their
operands are already laid out the way WMMA wants, so no runtime transpose is needed there).

### 1.6 `sched_group_barrier` patterns

The clearest statement of *why* this is needed: "LLVM single-buffers LDS and drains before every
WMMA unless you tell it not to. `__builtin_amdgcn_sched_group_barrier` is the fix, and the
granularity of the groups matters far more than their contents — coarser is better, up to the point
where the pattern asks for more outstanding loads than the hardware can hold"
(`README.md:138-141`).

- `r4d_gdn_chunk_scan_k128_v128_c64_bf16.hip:598-617` (`SCHED==5/6/7`): a `GRAN`-step group
  (2/4/8 k-steps) of `(VMEM read, LDS reads, WMMAs)` triples, explicitly chosen coarse over the
  finer per-step form (`SCHED==0..4`) because "the group barrier PINS the order at every boundary,
  so a per-step pattern over-constrains a loop the scheduler can already interleave." `SID`
  optionally tags each of the four WMMA-dense regions (main loop, V' loop, O loop, state-update
  loop) with its own `syncid` so the four patterns cannot bleed into each other when the compiler
  places their instructions in the same scheduling region (`:200-221`).
- `r4d_attn_prefill_h256_gqa6.hip`/`r4d_attn_decode_h256_gqa6.hip`: `PF`/`SGB` — prefetch `PF`
  (2/4/8) K or V fragments ahead of the matching WMMAs with a hard `sched_barrier()` between the
  load group and the WMMA group, "without the barrier PF is INERT" (`r4d_attn_prefill_h256_gqa6.hip:30-31`).
- `lds_barrier()` (a manually workgroup-scoped `local`-address-space fence, `r4d_common.h:63-74`)
  replaces every `__syncthreads()` in the GDN and attention kernels where the barrier only orders
  LDS traffic — a plain `__syncthreads()` on gfx12 emits `global_inv scope:SCOPE_SE` (a full vector
  cache invalidate) on both sides, which "throws away the L0/L1 lines the next tile's KV loads would
  have hit" (`r4d_common.h:65-69`). Same fix restated for the store-then-barrier case in
  `r4d_gdn_chunk_scan_k128_v128_c64_bf16.hip:38-44`: `__syncthreads()` after a global O store
  compiles to `s_wait_storecnt_dscnt 0x0`, which makes an LDS-only rendezvous also wait for HBM
  stores.

### 1.7 fp8 conversion paths

Two distinct paths, used for two different reasons:

1. **fp8-e4m3 KV → 16-bit widening** (attention): `__builtin_amdgcn_cvt_pk_f32_fp8` widens two fp8
   lanes to f32, then `DT16::pkv()` re-narrows to bf16/f16 via the same pack primitives as §1.2
   (`r4d_dt16.h:37-40,58-61`, driven through `fp8x8_to_16x4w()` at `:71-80`). This is a generic
   "any fp8 becomes any 16-bit format" path, reused unchanged for both bf16 and f16 targets.
2. **MXFP4 (e2m1) → e4m3, with per-block exponent folding** (GEMM): does **not** use the convert
   builtin above at all. Instead a 16-entry `__constant__` lookup table `kMag[16][2]` holds the
   eight e2m1 magnitudes pre-scaled by `2^-d` for every possible exponent difference `d` (0..15),
   as packed e4m3 bytes; unpacking a weight byte is nibble-split + two `v_perm_b32` table lookups +
   sign-bit OR (`r4d_gemm_mxfp4a8_nt_m64.hip:104-129`). The table deliberately runs into e4m3
   **subnormals** down to `2^-9` rather than flushing at the smallest normal `2^-6`: flushing early
   was **measured** to hit "6.7% of the weights on 36 output channels, for per-channel errors up to
   18.7%" on the real checkpoint, and extending the table is only correct because "the gfx12 fp8
   WMMA HONOURS e4m3 subnormals rather than flushing them — verified on hardware, exact for every
   `m*2^-9`, `m=1..7`" (`:24-31`).

### 1.8 int8 dot / WMMA paths

Two distinct int8 mechanisms, not to be conflated:
- `v_dot2_f32_bf16` (`__builtin_amdgcn_fdot2_f32_bf16`) — a **packed-bf16 dot product**, not WMMA,
  used only in the MoE-router GEMM `r4d_gemm_bf16_nt_m16.hip:64-69` where M<=16 makes per-scalar
  accumulation viable and WMMA's 16-row minimum would waste the tile.
- `v_dot2_f32_f16` (`__builtin_amdgcn_fdot2`) — used only inside the attention kernels' `DOT2`
  option to compute the softmax row-sum directly off the packed f16 P fragment ("4 instructions for
  8", `r4d_attn_prefill_h256_gqa6.hip:32-33`, applied at `r4d_attn_decode_h256_gqa6.hip:312-317`).
- `v_wmma_i32_16x16x16_iu8` — the actual int8 **matmul** path, used only in `r4d_gemm_w4a8_nt_m64`
  (§1.1). Its dequant is "two ands and a shift, no zero point" priced at "under 0.2% [of kernel
  time] at M=8" up to "5.1% at gate_up... for M=64" (`r4d_gemm_w4a8_nt_m64.hip:112-119`).

### 1.9 Memory-scope fences

Covered in §1.6 (`lds_barrier()`). One more fence worth flagging: the GDN chunk-scan's per-chunk
state writeback (`SWRITE()`) is followed by a barrier with an explicit correctness note — "no
barrier: `s.S` aliases nothing, and barrier (2) already proves every wave finished its U/O reads of
`s.S` this chunk" is the *removed* barrier at one point, contrasted with `BAR()` (4) which *is*
needed after the writeback for the next chunk's readers (`r4d_gdn_chunk_scan_k128_v128_c64_bf16.hip:845-848`)
— i.e. the author reasons about which barriers are redundant given the states already proven by
prior barriers in the same kernel, rather than syncing defensively everywhere.

### 1.10 "M<=64 skinny" design rationale, and where each kernel crosses from bandwidth- to compute-bound

The skinny-GEMM family's whole premise, stated most directly in `r4d_gemm_w4a16_nt_m64.hip:6-8`:
"at M=8 a projection moves M*K*2 bytes of activation against N*K bytes of weight, 82 KiB against 44
MiB on the drafter's MLP, so the format of the WEIGHT is the only thing in the cost model." At
small M the activation is negligible and the whole kernel is a weight-streaming problem; WMMA's
16-row minimum tile means the *activation* re-read grows 4x from M=16 to M=64 relative to the
weight (`r4d_gemm_w4a16_nt_m64.hip:36-43`, motivating `NPW` — column-tiles-per-wave — as the lever
that amortizes one activation-fragment load over several weight-fragment consumers).

Where each kernel actually crosses from bandwidth- to compute-bound, **measured**:
- `r4d_gemm_w4a16_nt_m64` (f16xf16 WMMA, 207 TF/s): at M<=32, "620-648 GB/s, 2.0x the block-fp8
  path" — DRAM-roofline bound, and "an ablation that deletes the unpack entirely moves M=8 by
  0.25%" (arithmetic is free there). At M=64 it flips fully compute-bound: "1.00-1.03x, 72-78 TF/s"
  against the 207 TF/s issue ceiling, i.e. the kernel is running at the matrix unit's actual rate,
  and "deleting the unpack only reaches 82-86" — the unpack costs 11.5-18.0% of the kernel there
  (`r4d_gemm_w4a16_nt_m64.hip:20-27,108-113`).
- `r4d_gemm_w4a8_nt_m64` (iu8 WMMA, 407 TF/s) exists *because* w4a16 hits that compute wall at
  M=64 — same weight bytes, 8-bit matrix instruction instead of the "slow" f16 one, 1.5-1.9x there
  (`r4d_gemm_w4a8_nt_m64.hip:1-16`).
- `r4d_gemm_mxfp4a8_nt_m64` (fp8 WMMA, 412 TF/s): the >100% "roofline" reading the file itself
  flags as a red herring — "a 47 MB buffer streamed in a loop is Infinity-Cache resident on this
  part, so treat the absolute fraction as a ranking signal between kernels reading the same bytes,
  not as achieved DRAM bandwidth" (`r4d_gemm_mxfp4a8_nt_m64.hip:81-84`) — i.e. even the "bandwidth"
  numbers reported for these kernels are partly Infinity-Cache-resident, not true HBM traffic, at
  this model's weight sizes.
- `r4d_gdn_kkt_solve_k128_c64_bf16`: explicitly memory-wait bound ("~2000 instructions per wave
  against its wall clock" — occupancy is the only lever, `r4d_gdn_kkt_solve_k128_c64_bf16.hip:106-108`).
- `r4d_gdn_recurrent_update_k128_v128_bf16_fp32state`: explicitly the opposite of the GEMMs —
  "THE STATE TRAFFIC IS THE WHOLE COST... at width 9 that is 113 MB per layer per rank, at the
  bandwidth limit of this part... a port that removes a Triton dependency and a kernel launch, not
  an arithmetic win" (`:14-22`).

### 1.11 Activation-quant formats

- **int8, per-row symmetric** (feeds `r4d_gemm_w4a8_nt_m64`): `r4d_quant_act_i8.hip` — one
  workgroup per row, absmax pass then a second pass that writes bytes **already reordered into the
  A-fragment byte order** the GEMM's `fragA8` wants (`r4d_quant_act_i8.hip:8-13,44-52`), so the
  GEMM's activation load is one `global_load_b64` per fragment instead of two strided
  `global_load_b32` (the file states the strided-plane alternative was tried and measured slower,
  106.7us vs 96.5us at M=64 — same note duplicated in `r4d_gemm_w4a8_nt_m64.hip:54-60`).
- **fp8-e4m3, per-row, plus a per-row reference exponent `wref`** (feeds
  `r4d_gemm_mxfp4a8_nt_m64`): the activation side is plain per-row e4m3 quant (`scale =
  max(1e-8,absmax)/448`, same convention r4dx's own `QuantActFp8Kernel` copies — see §2). The
  `wref` term is **weight-side**, not activation-side: `Wref[n] = max_k E8M0[n][k]`, computed
  offline, and is what makes the mxfp4 unpack a table lookup instead of a per-element rescale
  (§1.7). r4dx's own kernel comment (`src/kernels/include/r4dx/kernels/kernels.h:75-85`) correctly
  documents that its fp8 quant kernel has **no WMMA-fragment byte reorder** unlike the int8 path —
  "plain row-major byte order... unlike `r4d_quant_act_i8`'s int8 path, there is no WMMA-fragment
  byte reorder on the fp8 activation side" — verified against the mxfp4 GEMM's own `A + r*K + kg +
  mhalf` addressing (`r4d_gemm_mxfp4a8_nt_m64.hip:180`), which reads A row-major directly.

### 1.12 TODO / limitation notes the authors left

- **"REFUTED, and worth not trying again"** (`r4d_gemm_w4a16_nt_m64.hip:119-124`): folding the
  group scale into the B fragment via one `v_pk_mul_f16` was tried, measured *slower* on 3 of 4
  drafter shapes (qkv +1.2%, o +3.5%, down +5.5%, gate_up -2.3%) because the multiply lands inside
  the per-fragment dependent chain where the per-group epilogue it replaces sat outside it — "this
  kernel is stall-bound, not register-bound."
- Deleted-not-flagged options in the attention kernels, stated as a design principle: "Everything
  that measured far off the optimum was deleted rather than left behind a flag: the two-term-e4m3
  query and P legs, the single-term fp8 P leg (16% faster, 15x less accurate), the paired 8-bit LDS
  read that only those needed, the S-split, and a layout permutation that CONTIG made redundant"
  (`r4d_attn_prefill_h256_gqa6.hip:5-9`).
- `PFIN`/`APRE` in the chunk-scan kernel: both explicitly measured slower and "Left at 0" —
  interleaving prefetch loads inside the main WMMA loop (206.4us vs 196.2us) and hoisting the V'
  loop's A fragments before the main loop (255.7us vs 220.5us,
  `r4d_gdn_chunk_scan_k128_v128_c64_bf16.hip:155-177`).
- `SASG`/tile-shape alternative in the state update: measured 237.6us vs 196.7us, "left at 0"
  (`:82-93`).
- `ABL` ladders in both `chunk_scan` and `kkt_solve`: every nonzero value is stated to "produce
  WRONG results" and "exists solely to price one stage" — kept in the source as a reproducible
  cost-attribution tool, not dead code (`r4d_gdn_chunk_scan_k128_v128_c64_bf16.hip:178-186`,
  `r4d_gdn_kkt_solve_k128_c64_bf16.hip:63-69`).
- No proper prefill GEMM kernel exists yet — the README's kernel table and `docs/status.md`'s
  "Known gaps" both say so explicitly (`docs/status.md:144-145`); this is deliverable (d) below.
- `src/kernels/include/r4dx/kernels/kernels.h:64-67`: a TODO for multimodal mrope
  (`r4dx_rope_partial_mrope_multimodal_bf16`), text-only rope is what ships.

## 2. r4dx's own-kernel audit (`src/kernels/src/r4dx_kernels.hip`)

None of r4dx's nine own kernels use any of libr4d's hardware tricks: no WMMA, no bf16 packing via
`v_perm_b32`, no `sched_group_barrier`, no NT loads, no LDS staging beyond a trivial 8-float
cross-warp reduction buffer. Every one is a **plain grid-stride scalar loop**, `kThreads=256`, one
workgroup per row for the row-reduction kernels (matching `r4d_quant_act_i8.hip`'s own convention,
per the file's header comment). Specifically:

| kernel | launch shape | vectorization |
|---|---|---|
| `RmsNormKernel` / `ResidualRmsNormKernel` | `dim3(rows)`, `dim3(256)` | scalar `__bfloat162float` per element, `BlockReduceSum` via `__shfl_xor` (no packed bf16 math) |
| `ResidualAddKernel` | `dim3(ceil(n/256))`, `dim3(256)` | one element per thread, scalar |
| `SiluMulKernel` | `dim3(rows)`, `dim3(256)` | scalar, no `v_pk_*` |
| `RopePartialMropeKernel` | `dim3(T)`, `dim3(min(rotary_dim/2,256))` | scalar per (head, freq-bin); accurate `sincosf`, not the fast intrinsic (deliberate accuracy choice, `r4dx_kernels.hip:153-159`) |
| `QuantActFp8Kernel` | `dim3(M)`, `dim3(256)` | scalar absmax + scalar `FloatToFp8E4M3` per element |
| `KvWritePagedFp8Kernel` | `dim3(T,kv_heads)`, `dim3(min(head_dim,256))` | scalar |
| `ArgmaxKernel` | `dim3(1)`, `dim3(256)` | one-block grid-stride scan + shuffle-reduce, correctly int32-typed for `__shfl_xor` |

None of this is wrong for what these kernels are (row reductions and elementwise maps over at most
a few hundred KB at decode M=1), but it means:
- `RmsNormKernel`'s bf16 loads are one `__hip_bfloat16` at a time rather than a `ushort4`/`uint4`
  vector load the way every libr4d kernel loads its 16-bit operands (contrast with
  `r4d_gdn_gated_rmsnorm_h128_bf16.hip:43,53`, which loads 4 channels/lane as one `ushort4`).
- No kernel here uses `f2bf4`/`f2bf2` or any packed-pair convert, even though every store is a
  bf16 round — each element pays its own scalar `__float2bfloat16` (the built-in intrinsic path,
  not necessarily libr4d's own faster `f2bf`, though functionally equivalent).
- `RopePartialMropeKernel` loops `h` (heads) and `hd` (freq-bin) with `head_dim`-strided scalar
  loads/stores per head — 24 query heads x 1 rotated-pair load/store per thread iteration, no
  attempt to batch heads into a single vector load the way `r4d_gdn_conv_w4_h128_bf16.hip` batches
  4 channels/lane.

This is consistent with these being genuinely small, launch-overhead-bound kernels at decode
(`docs/perf.md`'s host_enqueue figures, §4 below) rather than a missed optimization — but it means
none of libr4d's hard-won per-instruction lessons have propagated into `src/kernels` yet.

## 3. (a) Hardware feature usage table

| hardware feature | used by libr4d? | used by r4dx (own kernels)? | opportunity |
|---|---|---|---|
| WMMA (bf16/f16/fp8/iu8, 16x16x16) | yes, extensively (§1.1) | no — r4dx's own kernels are all scalar/reduction, not matmul | none needed; r4dx's GEMMs already go through libr4d |
| iu4x iu4 WMMA (16x16x32, 830 TF/s) | named as a reference number only, never issued | no | real gap — see §5, a native-4-bit-weight WMMA prefill path is unexplored |
| bf16 software pack (`f2bf`/`f2bf4`, 2 adds + `v_perm_b32`) | yes, pervasive | no — scalar `__float2bfloat16`/`__bfloat162float` everywhere | low priority: r4dx's own kernels move tens of KB at decode M=1, launch-overhead bound not VALU bound (§2) |
| `v_cvt_pkrtz_f16_f32` (2-for-1 f32→f16) | yes, attention's `DT16<1>` (§1.1) | no (no f16 path in r4dx's own kernels at all) | n/a — r4dx has no f16 activations of its own |
| `global_load_tr_b128` (transpose load) | yes, attention V-fetch only (§1.5) | no | n/a — no transpose-shaped load in r4dx's own kernels |
| `sched_group_barrier` | yes, GDN + attention (§1.6) | no | n/a for kernels this small; would matter if r4dx ever wrote its own WMMA kernel |
| workgroup/local-scoped `lds_barrier()` (avoids SCOPE_SE cache invalidate) | yes, GDN + attention (§1.6/1.9) | no — r4dx's own kernels use plain `__syncthreads()` (`BlockReduceSum`/`BlockReduceMax`, `r4dx_kernels.hip:22-36,38-52`) | small, real: `ResidualRmsNormKernel`'s two `__syncthreads()` calls (`:100,101` implicitly via `BlockReduceSum`) invalidate the vector cache on every call for a barrier that only orders the shared `red[]`/`total` scratch — swapping in `lds_barrier()` is a drop-in, same shape as `r4d_common.h:70-74` |
| split-K in-block LDS reduction | yes, all 4 GEMM families (§1.4) | n/a (r4dx calls libr4d's GEMMs, doesn't reimplement) | n/a |
| NT (non-temporal) loads | yes, weight tiles in w4a16/w4a8 (§1.4) | no | n/a for r4dx kernels (nothing is a weight-streaming loop); note the model's own GEMM tuning sweep never swept `NT` either (`docs/perf.md:150-153`) — real, cheap follow-up for `tools/profile/tune_gemm.py` |
| fp8 WMMA + folded-exponent unpack table | yes, mxfp4 GEMM only (§1.7) | n/a | n/a |
| int8 WMMA (iu8) | yes, w4a8 GEMM (§1.1/1.8) | n/a | n/a |
| `v_dot2_f32_bf16`/`v_dot2_f32_f16` (packed dot, non-WMMA) | yes, router GEMM + attention softmax sum (§1.8) | no | none obviously applicable — r4dx's own kernels don't have a dot-product-shaped inner loop |
| device-side quantization fused with a producing kernel | no — libr4d's own quant kernels (`r4d_quant_act_i8`) are always standalone, one-workgroup-per-row (same as r4dx's `QuantActFp8Kernel`) | no | real, biggest opportunity in this audit — see §5 (rmsnorm→quant, silu_mul→quant fusion) |
| fused residual+norm in one kernel | **yes** — `r4d_gdn_recurrent_update` folds the gated-RMS-norm into its own decode epilogue specifically because prefill's chunk-scan workgroup only owns half a row (§1.10/architecture) | **kernel exists** (`r4dx_residual_rmsnorm_bf16`, `r4dx_kernels.hip:78-109`) **but is never called** anywhere in `src/model` (verified: only definition + declaration, no call site) | free win, zero new kernel code — see §5 item 1 |

## 4. (b) Launch count per decode token

Counted by reading the call sequence in `src/model/gdn_layer.cpp`, `src/model/mlp.cpp`,
`src/model/attention/include/r4dx/model/attention/attention_layer.hpp`, `src/model/linear.cpp`
(`ApplyLinear`'s per-layout dispatch), and `src/model/final_lm_head.cpp`, for one decode step
(T=1, M=1 per `ApplyLinear` chunk since M<64 never splits). `r4d_attn_decode_*` counts as **two**
kernel launches (the split-KV kernel itself plus `r4d_attn_splitkv_combine_kernel`,
`r4d_attn_paged_h256_gqa6.hip:130-136`). Embedding is a host-side gather + one async H2D memcpy
(`src/model/embedding.h:29-30`), not a device kernel launch, so it does not add to this count.

**Per GDN layer** (48 of 64 layers), decode path (`GdnLayer::Forward`, `is_prefill=false`):
rmsnorm(1) + in_proj_qkv GEMM(1, +1 quant/cast if quantized) + in_proj_a GEMM(1) + in_proj_b
GEMM(1) + in_proj_z GEMM(1) + GdnConvUpdate(1) + GdnRecurrentUpdate(1, norm fused) + out_proj
GEMM(1, +1 quant/cast if quantized) + residual_add(1) = **9 launches (bf16 layout), 11 launches
(w4a16/w4a8/mxfp4)**.

**Per attention layer** (16 of 64 layers): rmsnorm(1) + qg GEMM(1 — currently forced bf16 for
every `--layout`, `docs/perf.md`'s "Known limitation") + split_qg(1) + k GEMM(1) + v GEMM(1) +
q_norm(1) + k_norm(1) + rope(1) + kv_write(1) + attn decode(2) + gate_mul(1) + o GEMM(1, bf16) +
residual_add(1) = **14 launches**, currently layout-independent since qg/o are bf16-only today.

**Per MLP block** (all 64 layers): rmsnorm(1) + gate_up GEMM(1, +1 quant/cast if quantized) +
silu_mul(1) + down GEMM(1, +1 quant/cast if quantized) + residual_add(1) = **5 launches (bf16), 7
launches (quantized)**.

**final_norm+lm_head** (once): rmsnorm(1) + lm_head GEMM(1, +1 quant/cast if quantized) + widen(1)
= **3 (bf16), 4 (quantized)**.

**Total kernel launches per decode token**:

| layout | GDN (48x) | Attn (16x) | MLP (64x) | final | **total** |
|---|---|---|---|---|---|
| bf16 | 432 | 224 | 320 | 3 | **979** |
| w4a16 / w4a8 / mxfp4 | 528 | 224 | 448 | 4 | **≈1204** |

**Host overhead implied**: `docs/perf.md`'s per-op profile reports `host_enqueue` directly:
mxfp4 4.25ms (9.6% of a 44.28ms wall step), w4a16 2.48ms (6.5% of 38.13ms), w4a8 2.69ms (7.0% of
38.39ms), bf16 6.81ms (0.9% of 737.42ms) (`docs/perf.md:81-127`). Dividing by the launch counts
above gives an **inferred** per-launch host cost of ≈2.1-3.5 us for the quantized layouts
(2.48-4.25ms / ~1204 launches) — a plausible `hipLaunchKernelGGL` enqueue cost, and consistent
across all three quantized layouts despite their very different GPU-side kernel mixes, which is
evidence the ~1204 count (not GPU work) is the right explanatory variable for `host_enqueue`. bf16
does not fit the same line (6.81ms / 979 ≈ 6.95 us/launch, roughly 2-3x higher per launch than the
quantized layouts) — **inferred, not explained by anything read in this pass**; possibly larger
per-launch kernel-argument marshaling for the bf16 GEMM's plain-pointer ABI, or measurement noise
at bf16's much larger 737ms wall step. Flagged as an open question rather than asserted.

At 90-99% of wall time spent in `finish_wait` (the blocking final sync + logits readback, which is
GPU time the host is waiting on, not extra host work — `docs/perf.md:136-140`), `host_enqueue` is
not today's bottleneck in any layout; the fusion opportunities in §5 matter more for the GPU-side ms
they remove (fewer HBM round-trips of small intermediates) than for the enqueue time.

## 5. (c) Fusion opportunities

Estimated launch-count and time savings per decode token, using the ≈2.5 us/launch host-overhead
figure from §4 as a floor (**inferred**: actual GPU-side savings are likely larger than this,
since each eliminated kernel is also an eliminated HBM round-trip of its intermediate, but no
sub-kernel GPU timing exists in `docs/perf.md` to quantify that half — the profile is block-level
only, "block-level, not sub-kernel" per that file's own item-3 note).

1. **residual_add → rmsnorm fusion (free — the kernel already exists, just isn't called).**
   `r4dx_residual_rmsnorm_bf16` (`src/kernels/src/r4dx_kernels.hip:78-109`,
   `src/kernels/include/r4dx/kernels/kernels.h:26-31`) computes exactly `out_residual = x+residual;
   out_normed = rmsnorm(out_residual)*(1+w)` in one launch. Every GDN/attention layer boundary
   currently does `residual_add(x, sublayer_out) → x'` then the *next* stage's `rmsnorm(x')`
   as two separate kernels over the same `[T,hidden]` buffer: `GdnLayer::Forward`'s
   `r4dx_residual_add_bf16` (`gdn_layer.cpp:134-135`) immediately followed by `Mlp::Forward`'s
   `r4dx_rmsnorm_bf16` (`mlp.cpp:16-18`) reading the same output, and the symmetric case at the next
   layer's input rmsnorm reading MLP's own residual_add output. That is 2 fusable boundaries per
   decoder layer x 64 layers = **128 launches eliminated**, ≈**320 us/token** at the §4 floor —
   8-13% of the current 2.48-4.25ms `host_enqueue` budget in the quantized layouts, for zero new
   kernel code.
2. **rmsnorm → activation-quant fusion** (quantized layouts only). GDN's `in_proj_qkv` and MLP's
   `gate_up` both feed `ApplyLinear` directly off a freshly-computed `x_normed`
   (`gdn_layer.cpp:42-48`, `mlp.cpp:15-21`); `ApplyLinear` then issues its own separate quant/cast
   kernel (`r4dx_model_cast_bf16_to_f16` for w4a16, `r4dx::core::r4d::QuantActI8` for w4a8,
   `r4dx_quant_act_fp8e4m3_row` for mxfp4 — `linear.cpp:106-129`) before the GEMM. Folding the
   row-max/quantize pass into the rmsnorm kernel's own epilogue (rmsnorm already computes a
   per-row reduction over the same row) removes one full kernel launch and one HBM round-trip of
   `x_normed` per instance: 48 (GDN) + 64 (MLP) = **112 launches**, ≈**280 us/token**.
3. **silu_mul → activation-quant fusion** (quantized layouts, MLP `down` input). `SiluMulKernel`
   writes `h[T,intermediate]` (`mlp.cpp:23-25`), immediately consumed by `ApplyLinear`'s own quant
   kernel before `down`. 64 instances, ≈**160 us/token** at the launch-count floor — and unlike
   item 2, this one's HBM savings are the largest of the four: `h` is `[T,17408]` bf16, 34 KB at
   decode (T=1) but **34.8 MB per layer at a 64-row prefill chunk**, so at prefill (where MLP GEMMs
   are already 51-60% of GPU time, `docs/perf.md:129-133`) the removed round-trip is likely to
   matter more than the launch count alone suggests.
4. **GDN gated-rmsnorm → activation-quant fusion** (`out_proj` input). `out_core` is produced by
   `r4d_gdn_gated_rmsnorm_h128_bf16` at prefill (`gdn_layer.cpp:103-105`) or by
   `GdnRecurrentUpdate`'s already-fused epilogue at decode (`:123-128`) — **either way**,
   `ApplyLinear(out_proj, out_core, ...)` still issues its own separate quant kernel afterward, so
   this fusion opportunity exists identically at both decode and prefill. 48 instances,
   ≈**120 us/token**.
5. **rope → kv-cache-write fusion.** `r4dx_rope_partial_mrope_bf16` rotates `k` in place
   (`attention_layer.hpp:130-133`), then `r4dx_kv_write_paged_fp8_hnd` re-reads that same `k`
   buffer (plus `v`, untouched by rope) to quantize and scatter into the paged cache
   (`:137-141`). Since rope only ever needs `k`'s two halves per (head, freq-bin) once, its store
   could write directly into the fp8 cache slot instead of back to the bf16 `k` buffer, deleting
   one launch per attention layer: 16 instances, ≈**40 us/token**. Smallest of the five, but the
   two kernels already iterate the same `[Hkv, head_dim]` shape per token, so the fusion is
   mechanical.
6. **GDN control uploads (named in the task brief) — already fixed, no remaining opportunity.**
   `docs/perf.md`'s review-fix pass already replaced per-call `UploadArray` (3 host-blocking
   `hipStreamSynchronize` calls x 48 GDN layers = 144 syncs/token) with `GdnControlCache`, which
   uploads each distinct `(T,slot)` control array once ever (`docs/perf.md:455-462`,
   confirmed live in `gdn_layer.cpp:66-67,121`). Nothing further to fuse here; flagged as closed so
   it doesn't get re-proposed.

**Total estimated**: items 1-5 remove ≈**368 launches/token** (≈**920 us** at the §4 floor) from
the quantized layouts' current ~1204, a ~31% cut in launch count. Since `host_enqueue` is 6.5-9.6%
of wall time today, this alone would not directly move tok/s by 31% — but items 2-5 also remove
real HBM round-trips of intermediates the current code writes and immediately re-reads, so the
actual GPU-time saving (not measurable from the existing block-level profile) is plausibly larger
than the launch-count floor, especially at prefill chunk sizes (T up to 64) where those
intermediates are 64x larger than at decode.

## 6. (d) What a proper prefill kernel should look like

`docs/status.md`'s "Known gaps" and `docs/architecture.md`'s "Interim chunked prefill" both flag
this as explicit future work: today a prompt is chunked into <=64-row pieces and run through the
same skinny `r4d_gemm_*_nt_m64` kernels decode uses, which `docs/perf.md`'s own GEMM-tuning-sweep
section confirms is leaving real throughput on the table (prefill improved 38-85% just from
retuning `WV/SK/MB/NPW` on the *existing* skinny kernel, `docs/perf.md:168-176` — a dedicated
kernel should do better still). The following is design **inferred** from libr4d's own proven
techniques, not a measured recommendation — no prefill-kernel code exists yet to measure.

**The premise that must change.** Every current skinny GEMM is built on "the weight is read once
per k-step and never reused, so streaming it (NT loads where they win) and never staging it in LDS
is correct" (§1.10, §1.4). That premise holds exactly because M is small enough that the N-tile x
K-step space is what needs filling, not because it's true GEMM design. At real prefill M (hundreds
to low thousands of rows per chunk, not capped at 64), the FLOPs-per-weight-byte ratio flips: a
16x16 weight tile, dequantized once, can now feed `M/16` row-tiles of WMMA instead of 1-4. A
"proper" prefill kernel is therefore not a wider version of the skinny kernel — it is a
classically-tiled GEMM (2D M×N block tiling, both operands staged through LDS, weight dequantized
into LDS once per tile and reused across many row-tiles) that happens to reuse every low-level
trick the skinny kernels and the GDN chunk-scan already proved on this hardware:

- **Fragment layout and weight pre-permutation carry over unchanged.** The
  `idx=lane%16, k=8*(e>>2)+4*(lane>>4)+(e&3)` convention (§1.1) and the offline
  fragment-order weight permutation both `r4d_gemm_w4a8_nt_m64` and `r4d_gemm_mxfp4a8_nt_m64`
  already do (`r4d_gemm_w4a8_nt_m64.hip:22-34`, `r4d_gemm_mxfp4a8_nt_m64.hip:48-59`) are
  layout-agnostic — they describe how one WMMA tile's operands sit in registers, independent of
  block size. A prefill kernel should keep this exact packer/permuter and just consume more tiles
  per weight load.
- **Pick the matrix instruction the checkpoint format already implies, exactly as the existing
  kernels do**: iu8 WMMA (407 TF/s) for w4a8-style int4-weight/int8-activation checkpoints, fp8
  WMMA (412 TF/s) for native MXFP4 checkpoints — both already measured at effectively the same
  issue rate (§1.1), so the *format* decides, not a speed argument. The dequant paths to reuse are
  `dequant8()` (two ANDs + one shift, `r4d_gemm_w4a8_nt_m64.hip:115-119`) and
  `r4d_mxfp4_unpack8()` (the folded-exponent table lookup, `r4d_gemm_mxfp4a8_nt_m64.hip:119-129`)
  — both proven to cost under a few percent of kernel time even at M=64
  (`r4d_gemm_w4a8_nt_m64.hip:112-114`).
- **The unexplored native path**: `v_wmma_i32_16x16x32_iu4` (830 TF/s, **measured** as a reference
  number but never issued by any kernel, §1.1) doubles the iu8 rate. Using it directly on the
  4-bit weight (skipping the dequant-to-int8 step entirely) would need int4 *activations* too,
  which no kernel here has attempted or validated for accuracy — **flagged as an open question**,
  not a recommendation, since nothing in this codebase establishes whether int4 activations are
  accurate enough for this model. A safer half-step worth prototyping: keep int8 activations and
  check whether the ISA has an asymmetric iu4(weight) x iu8(activation) WMMA form at a rate between
  407 and 830 TF/s — **not confirmed anywhere in the sources read this pass**.
- **Split-K shrinks or disappears as M grows.** `SK` exists today specifically because "the
  independent N fragments... does not fill the part" at small M
  (`r4d_gemm_mxfp4a8_nt_m64.hip:41-44`) — once M is large enough that `(N/16) x (M/MT)` tiles alone
  saturate the CUs, the K-split's LDS-reduction machinery (and its associated non-determinism
  across SK choices, §1.4) can go away, simplifying the kernel and removing the
  rounding-order-changes-output-tokens hazard `docs/perf.md` already hit once.
- **Stage both operands through LDS with double buffering**, using the exact scheduling discipline
  `r4d_gdn_chunk_scan_k128_v128_c64_bf16` proved: workgroup-scoped `lds_barrier()` instead of
  `__syncthreads()` (§1.6/1.9), and a coarse `sched_group_barrier` grouping (`GRAN`=2-4 k-steps,
  §1.6) over `(global load, LDS read, WMMA)` triples rather than a per-step pattern. This is the
  same machinery the current skinny kernels don't need (they never stage the activation) but a
  real GEMM will.
- **NT loads flip roles.** Today NT applies to the weight because it's read once and never reused
  (§1.4). Once the weight is staged in LDS and reused across `M/16` row-tiles, the weight is
  exactly the operand worth keeping *temporal* (cache-resident) rather than NT-streamed — closer to
  `r4d_gemm_bf16_nt_m16`'s router-GEMM choice ("cache-resident by default, since the gate weights
  fit the 64 MB Infinity Cache", `r4d_gemm_bf16_nt_m16.hip:30-32`) than to the current skinny
  kernels' NT default.
- **Occupancy**: follow `r4d_gdn_kkt_solve`'s discipline of *measuring* forced occupancy rather
  than assuming higher is better — that kernel's own ablation showed forcing more waves/SIMD than
  the natural VGPR-driven count spills and is 1.57x slower (§1.3). A prefill GEMM at real M will
  have far more independent tiles to schedule than the skinny kernels do, so the natural-occupancy
  default is a safer starting point than copying the skinny kernels' `__launch_bounds__(1024)`
  (no minimum) verbatim.

## Sources

All file:line citations above are against this checkout: `third_party/libr4d/{README.md, r4d.h,
r4d_common.h, r4d_dt16.h, r4d_gdn_wmma.h, r4d_gemm_bf16_nt_m16.hip, r4d_gemm_bf16_nt_m64.hip,
r4d_gemm_w4a16_nt_m64.hip, r4d_gemm_w4a8_nt_m64.hip, r4d_gemm_mxfp4a8_nt_m64.hip,
r4d_attn_prefill_h256_gqa6.hip, r4d_attn_decode_h256_gqa6.hip, r4d_attn_paged_h256_gqa6.hip,
r4d_gdn_chunk_scan_k128_v128_c64_bf16.hip, r4d_gdn_kkt_solve_k128_c64_bf16.hip,
r4d_gdn_recurrent_update_k128_v128_bf16_fp32state.hip, r4d_gdn_conv_w4_h128_bf16.hip,
r4d_gdn_gated_rmsnorm_h128_bf16.hip, r4d_quant_act_i8.hip, r4d_dflash_conv_body.h,
r4d_dflash_conv_t2_g16_bf16.hip}`; `src/kernels/{src/r4dx_kernels.hip,
include/r4dx/kernels/kernels.h}`; `src/model/{gdn_layer.cpp, mlp.cpp, linear.cpp,
final_lm_head.cpp, gemm_tuning_table.inc, attention/include/r4dx/model/attention/attention_layer.hpp}`;
`docs/{perf.md, architecture.md, status.md}`. No GPU access this pass; every number labeled
"measured" is the libr4d/r4dx authors' own prior measurement quoted from their comments or
`docs/perf.md`'s tables, not independently re-verified here.
