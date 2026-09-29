# M = 256 trellis GEMM (prototype, bench only)

Status: a measured prototype on branch `linear`. Nothing in r4dx calls it: it is not in `R4D_UNITS`, not in
`ApplyLinear`, no shipped kernel, tuning row or default changed. Kernel `third_party/libr4d/
r4d_gemm_trellis_nt_m256.hip` (libr4d branch `linear`), bench `tests/kernels/tool_trellis_m256_bench.hip`
(`build_m256_bench.ps1`, hipcc, no CMake target).

## What it is

One launch computes M = 256 rows (or 128) of the trellis linear with the bits the shipped M = 64 kernel gives
each 64-row slice at the same (SK, SKG). Every accumulator keeps its k-ordered chain of 16-k WMMA; the SK
slices are summed in slice order from 0.f, the SKG partials in y order from 0.f, then the stock FWHT stages,
svh, out_scale and one bf16 rounding. Row tiles are independent in WMMA, so nothing else can differ.

* Decode shared through LDS (the "M8" idea, docs/trellis-kernel.md 4.7). A workgroup is RG x SK waves
  (RG = M / 64). Wave (rg, ks) owns rows 64 rg.. and K slice ks with the stock MT = 4 accumulator tile. Per
  step of U k-tiles the RG waves of one K slice each decode ONE of the slice's NP x U (tile pair, k-tile)
  blocks, store the two f16 fragments (ds_store_b128) to a double-buffered LDS slot, one workgroup barrier,
  then every wave reads all of them (ds_load_b128) and runs NP U 2 4 WMMA. Decode and weight loads are
  amortised over 256 rows. Shipped configuration: RG 4, NP 1, U 4 (Wc = 32 columns per block, 16 waves for
  SK 4), 174 VGPRs at KB 4 and 186 at KB 5, no scratch, LDS 32 KB staging / 48 KB reduction.
* Epilogue in the accumulator layout (all lane-contiguous b128, no transposes):
  (1) SK reduction: tile i is owned by slice i % SK; the other slices store their acc[.][.][i] to LDS, the
  owner sums in slice order taking its own term from registers; (2) the owner writes the sum to ws in lane
  order and takes a ticket (the stock scheme: the last of the SKG x 128/Wc contributors finishes, tickets
  reset themselves, no inter-block wait); (3) the last block y-sums the ws units, runs all seven FWHT stages
  in registers (lane bits 0-2, the f pair, lane bit 3, then the block index), scales and stores bf16.
* ws is SKG M N x 4 B (gate_up 35.6 MB, down 10.5 MB at M = 256); tickets N/128 words, as the stock kernel.

## Bit identity (MEASURED, `tool_trellis_m256_bench --mode verify`)

C[256][N] bf16 of one launch equals, byte for byte, four M = 64 launches of the shipped kernel on the same
inputs, against EVERY legal stock tuning of the same (SK, SKG) (WV 1/2/4, NP 1/2, U 1/2/4, NT 0/1: 24-25 of
them per shape, including the shipped table row), for all seven linear classes x KB 4/5, 6 seeds, with the
n_split (two activation matrices) pass, M = 128 against two launches (4 seeds). Zero differing bytes, tickets
reset every time. Negative control: against a stock tuning of a different SK the same comparison finds
206-13345 differing bf16 per shape (0.08-0.15%), so it does detect an accumulation-order change.
Coverage limit: random weights (any 32-bit words decode), random f16 A, M a multiple of 64.

## Speed (MEASURED, device 1, final_m256_run1/2.log, rounds 15, batch 6 weight copies)

Speedup of one M = 256 launch against 4 x the shipped M = 64 row on cold weights (each launch on a copy no
launch of the batch has read), median of per-round ratios; run 1 / run 2.

| class (layers) | KB 4 | KB 5 |
|---|---|---|
| mlp.gate_up (64) | 1.49 / 1.49 | 1.49 / 1.49 |
| mlp.down (64) | 1.70 / 1.71 (SK 4: shipped row is SK 16) | 1.70 / 1.66 |
| gdn.in_proj_qkv (48) | 1.63 / 1.58 (SK 2) | 1.66 / 1.66 (SK 2) |
| gdn.in_proj_z (48) | 1.58 / 1.56 | 1.56 / 1.57 (SK 8; SK 4: 1.71 / 1.77) |
| gdn.out_proj, attn.o (64) | 1.58 / 1.59 | 1.70 / 1.70 (SK 8; SK 4: 1.73 / 1.75) |
| attn.qg (16) | 1.45 / 1.49 | 1.53 / 1.52 |
| attn.k, attn.v (32) | 1.83 / 1.88 (SK 8) | 2.02 / 2.08 (SK 8) |
| **MLP pair** | **1.555 / 1.556** | **1.548 / 1.546** |
| all seven classes, layer-weighted | 1.564 / 1.562 | 1.587 / 1.572 |

The MLP pair against four M = 64 launches on the SAME weights (a Model chunk of 256 with today's kernel) is
1.519 / 1.493 (KB 4) and 1.551 / 1.532 (KB 5). Run-to-run spread of one shape is about +-3% (GPU state);
per-round pair ratios span 1.44-1.63. M = 128 (RG 2, 8 waves for SK 4): pair 1.13 (KB 4), 1.14 (KB 5);
all classes 1.13 / 1.15. Not worth a kernel of its own except for a 128-row tail.
Sum over the model's layers, bench-derived (DERIVED, excludes in_proj_a/b and the transforms): the seven
classes cost 0.70-0.72 (KB 4) and 0.76-0.78 (KB 5) ms/token at M = 64 and 0.45-0.46 / 0.48-0.49 ms/token at
M = 256 (two runs), i.e. about 0.25-0.28 ms/token saved (the probe's whole-chunk GEMM time is 46 ms / 64 rows
= 0.72 ms/token).

## Where the time goes (MEASURED with ablation builds, gate_up KB 4)

Loop alone 721 us, whole kernel 790 us: epilogue 69 us = SK reduction 14 + ws stores and ticket 15 + the last
block's finish 38. The loop's floor at 2.59 GHz is WMMA 552 + decode 68 = 620 us, so it runs at 86%. Removing
the barrier or the LDS reads changes nothing (the WMMA-only skeleton is 743 us); the non-temporal weight
loads change nothing. (Replacing decode or the loads by VALU stand-ins does not lower the time either, but
that ablation is not clean: the stand-ins cost about the VALU they replace.)

## What was tried and dropped

* (RG 4, NP 2, U 2): 0-15% slower than NP 1, U 4 on every class and it spills (24 B/lane at KB 4, 52-260 at
  KB 5). Removed.
* M = 128: above. 32-wave workgroups (SK 8): slower than SK 4 (z KB 5: 175 vs 159 us); kept only because
  the shipped KB 5 z / out rows are SK 8.
* Non-temporal weight loads: within noise.

## Not done / limits for whoever wires it

* mlp.down KB 4's shipped M = 64 row is SK 16 (WV1, SKG 2). 16 slices x 4 row groups is 64 waves, so the
  kernel cannot reproduce it: the M = 256 numbers use SK 4, which is bit-identical to the SK 4 stock tunings
  but not to today's KB 4 down row (the down GEMM is 5% slower at SK 4 in stock). Using it in production means
  re-pinning that row's M = 64 band to SK 4 (the bits of M = 33..64 chunks change once).
* Not wired: ApplyLinear still slices any M into 64-row launches; the Model chunk is still 64; attention,
  GDN, transforms and scratch at 256 rows are untested. Tail chunks would use this kernel at 128 and the M = 64
  kernel below that.
* The build's zero-scratch / 190-VGPR check (third_party/check_trellis_isa.cmake) does not cover this unit yet;
  compile report: every shipped instantiation is 151-190 VGPRs, 0 scratch.
* A-layout: fragment-contiguous activations (a prefill-only input-transform change, measured -5..-8% on the
  M = 64 kernel) were not tried here.
