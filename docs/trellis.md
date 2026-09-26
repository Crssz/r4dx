# EXL3 trellis quantization -- an implementable specification (oracle input)

Branch `quant2` (worktree `%USERPROFILE%\dev\r4dx-quant2`). Written 2026-09-26. This is the spec the
**EXL3 oracle** is built from: the KL of EXL3-style trellis quantization on Qwen3.8-27B and the rung-4
tokens (`tools/reference/kl_corpus/tokens_canon.json`), weights only, at 3.5 / 4.0 / 4.5 / 5.0 bpw.
Gate for writing any RDNA4 trellis kernel: mean KL <= ~0.008 at 4.0-4.5 bpw (today's quant2 recipe:
~0.0162 at ~4.5 bpw; Unsloth UD-Q4_K_XL: 0.0071 weights-only at ~5.05 bpw). Read the oracle's KL with
two caveats (13, 14): its Hessians are domain-matched to the KL corpus, so it is probably optimistic
against a stock EXL3 conversion and against UD-Q4_K_XL (fair against quant2, same Hessians); and
EXL3's own Hessian basis is measurably worse on attention q/k/v than the matched one our converter
would use, so the gate is judged on the matched-basis oracle.

Everything below describes exllamav3 **as of commit `6b84a21b6f1e5da3f291b9e1019061f0de788279`**
(2026-09-20, "MSVC fix (__builtin_popcount)"), MIT-licensed, (c) turboderp; the trellis/codebook
design comes from QTIP (Tseng, Sun, Hou, De Sa, arXiv 2406.11235). Where this doc says "EXL3 does X",
X was read from that commit's source. Where it says **(derived)**, it is my arithmetic or a
CPU check in this worktree, not a quote. Anything I could not pin down is in section 12, not guessed.

---

## 0. Summary

| Item | EXL3 (default `convert.py` settings) | Source |
|---|---|---|
| Tile | 16 x 16 weights (16 input rows k x 16 output cols n) = one 256-step trellis ring | Q `ldlq`, QTK `L`, QTIP A.2 |
| Trellis | bitshift trellis, state L = 16 bits, V = 1, K bits/weight, K in 1..8 | QTK `edges = 65536 >> K`, QCU range check |
| Fractional K | K + 0.5 (KA = floor K, per-position width `KA + ((0xAAAA >> (i&15)) & 1)`), mul1 only; allocator only emits 1.5/2.5/3.5 | Q `frac_k`, FR, QTF, AL `rate_floor` |
| Codebook | **mul1** (default): `x = state * 0x83DCD12D; v = fp16((1024 + bytesum(x)) * fp16(0x1EEE) + fp16(0xC931))` | CB, CM `--codebook` |
| Tail-biting | yes: two Viterbi passes (roll 128 unconstrained, then roll 0 pinned to the recovered overlap) | QTK orchestration, QTIP Alg. 4 |
| Stored per tile | exactly 256*K bits (16*K uint16): the low K bits of each position's 16-bit state, MSB-first; no per-tile scale, no start state | Q `pack_trellis`, PK, DQ |
| Viterbi objective | plain squared error in the rotated/scaled domain, fp16 path costs, ties to the smallest predecessor edge | QTK |
| Incoherence | random signs both sides + blockwise 128-point Sylvester Hadamard both sides (`had_k = had_n = 128`) | Q globals, `regularize` |
| Scales | per-input-channel `suh` (k fp16) and per-output-channel `svh` (n fp16), signs folded in; one global `g_scale` folded into `suh` | Q `regularize`, `quantize_exl3` |
| Rounding | block LDLQ, 16-row blocks, rows processed **last to first**, lazy spans of 128 rows, damping 0.025 x mean(diag H) | Q `finalize_capture_H`, `block_ldl`, `ldlq` |
| Post-polish | `refit_scales`: 2 rounds of closed-form refits of svh then suh in the H metric, trellis fixed | Q `refit_scales` |
| Hessian | `X^T X / rows` of each linear's input (finite rows only), **sequential** (inputs come from the already-quantized model), 250 x 2048 calibration tokens | LIN `capture_H`, CM, CD |
| Overhead | 16 (k + n) bits per linear (suh + svh), +4 bytes (codebook marker) -> +0.0045 bpw on this model | section 10 |
| Decode | ~4 int/fp ops per weight (mul1: extract, IMUL, DP4A, 1/2 PRMT, 1/2 HFMA2) + the MMA; 3 ops and no MMA on the int8-activation GEMV | DQ, CB, G8 |

---

## 1. Sources and conventions

### 1.1 Files (all at commit `6b84a21`)

| Tag | Path in `github.com/turboderp-org/exllamav3` | What it holds |
|---|---|---|
| Q | `exllamav3/modules/quant/exl3_lib/quantize.py` | the whole host-side quantizer |
| EX | `exllamav3/modules/quant/exl3.py` | runtime linear: tensors, dispatch |
| LIN | `exllamav3/modules/linear.py` | Hessian capture hook, `convert_exl3` |
| CB | `exllamav3/exllamav3_ext/quant/codebook.cuh` | state -> value decoders |
| QCU | `exllamav3/exllamav3_ext/quant/quantize.cu` | Viterbi host entry points, kernel choice |
| QTK | `exllamav3/exllamav3_ext/quant/quantize_tiles_kernel.cuh` | plain Viterbi kernel |
| QTO | `exllamav3/exllamav3_ext/quant/quantize_tiles_optimized.cuh` | optimized Viterbi kernel |
| QTF | `exllamav3/exllamav3_ext/quant/quantize_tiles_frac_kernel.cuh` | fractional-rate Viterbi |
| PK | `exllamav3/exllamav3_ext/quant/pack.cu` | integer-K bit packer / unpacker |
| FR | `exllamav3/exllamav3_ext/quant/frac.cu` | fractional-K packer / unpacker |
| DQ | `exllamav3/exllamav3_ext/quant/exl3_dq.cuh` | in-GEMM trellis decode |
| GC, GK, GI | `.../quant/exl3_gemm.cu`, `exl3_gemm_kernel.cuh`, `exl3_gemm_inner.cuh` | GEMM host, kernel, main loop |
| GV, GVK | `.../quant/exl3_gemv.cu`, `exl3_gemv_kernel.cuh` | QTIP-style small-m GEMV |
| G8, G8K | `.../quant/exl3_gemv_int8.cu`, `exl3_gemv_int8_kernel.cuh` | int8-activation GEMV |
| HI | `.../quant/hadamard_inner.cuh` | 128-point Hadamard device code |
| HPY | `exllamav3/util/hadamard.py` | host Hadamard matrices |
| CM | `exllamav3/conversion/convert_model.py` | conversion driver, args, quant_args |
| AL | `exllamav3/conversion/allocation.py` | bpw -> per-tensor K |
| CD | `exllamav3/conversion/calibration_data.py` | default calibration set |
| QC | `exllamav3/conversion/quant_config.py` | `quantization_config.json` writer |
| ARCH | `exllamav3/architecture/qwen3_5.py` | this model family's linears |
| DOC | `doc/convert.md` | converter CLI |
| QTIP | arXiv 2406.11235 | sections 2.1, 3.1, 3.1.1, 3.2, Tables 1-2, Appendix A.2 |

**Citation form.** `Q::regularize` means function `regularize` in file Q. Line numbers are **not**
given: the only access path allowed in this workflow (WebFetch, through a summarizing model; no
downloads) returned mutually inconsistent line numbers for the same file (e.g. `block_ldl` reported
at 348, 803 and 829 of Q), so none of them are trustworthy. Every statement quoted here is exact and
grep-able in a checkout of the pinned commit; pinning line ranges is open question Q1.

### 1.2 Notation

- EXL3 stores a linear as **W with shape (k, n) = (in_features, out_features)**, i.e. the HF weight
  transposed, and computes `y = x W` for a row vector x (Q::ldlq docstring: "Input weights, shape
  (k, n)").
- `P_128` = the 128 x 128 natural-order Sylvester Hadamard matrix scaled by 1/sqrt(128):
  `P[i][j] = (-1)^popcount(i & j) / sqrt(128)`. It is symmetric and orthogonal, so `P^-1 = P`.
  `P_k` / `P_n` = block-diagonal copies of `P_128` along k / n (requires k, n multiples of 128).
- `fp16(.)` = round-to-nearest-even to IEEE binary16.

---

## 2. Pipeline for one linear (default settings), in order

1. **Hessian** (section 7.1): `H = sum over finite calibration rows of x^T x`, x the input of this
   linear in the partially-quantized model; linears with the same input share one H.
2. **Finalize H** (7.2), once per shared H: `H /= rows`; `H += 0.025 * mean(diag H) * I`; draw input
   signs `s_u`; rotate `H_r = P_k S_u H S_u P_k`; block-LDL `H_r = L D L^T` with 16 x 16 blocks.
3. **Seed** `torch.manual_seed(module_index)`; draw output signs `s_v` (8.1).
4. **Regularize** (8.2): output-channel scales, divide, output Hadamard, input-channel scales,
   divide, input Hadamard, global scale search -> `W_r` (k, n) plus `su` (k), `sv` (n).
5. **Block LDLQ** (7.3) over 16-row blocks from the last to the first, each 16 x n row block cut
   into n/16 tiles, each tile Viterbi-quantized (section 6) on the feedback-corrected values.
6. **Un-rotate** `W_hat = diag(su) P_k Q P_n diag(sv)` (8.4).
7. **Refit** `su`, `sv` in the H metric, 2 rounds (7.5).
8. **Store** `trellis` (int16, (k/16, n/16, 16K)), `suh = fp16(su)`, `svh = fp16(sv)`, and a scalar
   codebook marker (`mul1` = 0x83DCD12D as int32) (Q::quantize_exl3, QC).

---

## 3. Tile geometry, ordering and trellis parameters

### 3.1 Tiles

- A tile is 16 consecutive input rows x 16 consecutive output columns of W (k, n). Tile `(tk, tn)`
  covers `W[16 tk : 16 tk + 16, 16 tn : 16 tn + 16]`. Tiles are stored **k-major**: the trellis tensor
  has shape `(k/16, n/16, 16*K)` uint16 (stored as int16), tile `(tk, tn)` at `[tk][tn][:]`
  (Q::pack_trellis `packed_shape = (shape[0], shape[1], trellis_words(K))`; GC arg doc "shape
  (k//16, n//16, 16*K), dtype uint16").
- Shape requirements: `size_k % 16 == 0`, `size_n % 128 == 0` (Q::ldlq asserts), and k, n multiples
  of 128 for the blockwise Hadamards (`had_k, had_n = 128, 128`; GV rejects `size_k % 128 ||
  size_n % 128`). All quantized linears of Qwen3.8-27B satisfy this (section 10).
- Within a tile, element `(r, c)` (r = k offset, c = n offset) has row-major index `r*16 + c`
  (Q::ldlq: `tiles = rows.reshape(16, tiles_n, 16).permute(1, 0, 2).reshape(tiles_n, 256)`).

### 3.2 The trellis walks the tile in tensor-core order

Before quantization each tile is permuted, `tiles = tiles[:, tensor_core_perm(device)]`; the
reconstruction is un-permuted, `quant_w = quant_w[:, tensor_core_perm_i(device)]` with
`tensor_core_perm_i = argsort(perm)`, while the indices are stored as returned, i.e. **in the
permuted order** (Q::ldlq `b_encoded[bi // 16 : bj // 16] = quant_i.unsqueeze(0)`). Sequence position
`p = 8 t + j` (t = 0..31, j = 0..7) holds row-major element `perm[p]` (Q::tensor_core_perm, exact):

```
r0 = (t % 4) * 2;  r1 = r0 + 1;  r2 = r0 + 8;  r3 = r0 + 9
c0 = t // 4;       c1 = c0 + 8
perm[8t+0..7] = r0*16+c0, r1*16+c0, r2*16+c0, r3*16+c0, r0*16+c1, r1*16+c1, r2*16+c1, r3*16+c1
```

So positions `8t .. 8t+7` are exactly the 8 B-operand values lane t needs for the two NVIDIA
`mma.m16n8k16` halves of the tile (k = 2(t%4) + {0,1} and +8, n = t/4 and t/4 + 8); the GEMM decodes
them with `t_offset = lane_id << 3` (GI `dq_dispatch<bits, cb, half_k>(shb, lane_id << 3, ...)`).
**(derived)** The permutation is a fixed bijection inside the tile; for incoherence-processed
(approximately i.i.d.) weights it does not change the expected distortion, so an RDNA4 format may
substitute its own WMMA-fragment order. The oracle can use EXL3's order.

### 3.3 Trellis parameters

- **State:** L = 16 bits, one weight per step (V = 1), T = 256 steps per tile (QTK comment: "L is the
  tile length (number of weights per tail-biting trellis ring). 256 = the 16x16 EXL3 tile"; QTIP 3.1,
  Table 10 motivates L = 16).
- **Bits per weight K:** integers 1..8 (QCU: "quantize_tiles K must be in range 1..8").
- **Fractional K** (Q::frac_k): K must be an integer or integer + 0.5 (`assert abs(ka + 0.5 - K) <
  1e-9`); K + 0.5 maps to `(KA, MASK) = (int(K), 0xAAAA)`. The Viterbi/packer support any 16-bit MASK
  with `16*KA + popcount(MASK)` even and KA in 1..7 (FR "KA must be 1..7", "bits per 16 weights must be
  even"), but only 0xAAAA is produced. Fractional rates are **mul1 only** (QTF comment: "mul1 codebook
  only: the fractional rates exist for new quants, the 3INST/MCG codebooks are legacy"; GC
  `TORCH_CHECK(!half_k || (tile_u16 % 16 == 8 && mul1))`).
- **Words per tile:** `trellis_words(K) = 16 * K` uint16 (must be integral), e.g. K = 2.5 -> 40.
  Runtime infers K from the last dim: `K = trellis.shape[-1] / 16`, `half_k = (tile_u16 % 16) != 0`
  (EX, GC, QC `bits_per_weight = shape[-1] / 16`).
- **Mixing:** no mixing of K inside a tensor except the fixed period-16 KA/KA+1 pattern of a
  fractional rate. Different tensors get different K by the allocator (section 9).

---

## 4. The procedural codebooks (state -> value)

The value of a position is a pure function of its 16-bit state s (zero-extended to 32 bits). Three
codebooks exist; the converter default is **mul1** (CM: `"--codebook", default = "mul1", help =
"Codebook: mul1 (default), mcg or 3inst"`). The stored marker tensor selects the codebook at load:
`<key>.mul1` or `<key>.mcg` holding the multiplier as int32; neither = 3INST (Q::quantize_exl3, QC).
All arithmetic below is exact 32-bit unsigned wrap-around integer math followed by fp16 ops with one
rounding each (CB).

### 4.1 mul1 (cb = 2, default)

```
x   = (s * 0x83DCD12D) mod 2^32                     # codebook_mul1_mult (Q), CB decode_3inst<2>
u   = 0x6400 + b0(x) + b1(x) + b2(x) + b3(x)          # __dp4a(x, 0x01010101u, 0x6400u); 0 <= sum <= 1020
h   = half_from_bits(u)                              # = 1024.0 + sum, exact (ulp is 1 on [1024, 2048))
v   = fp16( h * fp16(0x1EEE) + fp16(0xC931) )         # __hfma, one rounding
    = fp16( (1024 + sum) * 0.00676727294921875 - 10.3828125 )
    ~= 0.0067673 * sum - 3.453125
```

The pair form `decode_mul1_product_2` does the two dp4a's, packs the two 16-bit results into a half2
and does one `__hfma2` with `k_inv_h2 = 0x1eee`, `k_bias_h2 = 0xc931` (CB). This is QTIP's 1MAD idea
(sum of four uniform bytes ~ Gaussian; QTIP 3.1.1 Algorithm 1 uses `a = 34038481, b = 76625530` and
`(x - 510)/147.8`) with a pure multiply (no additive constant) and a fp16 affine map.

### 4.2 3INST (cb = 0, legacy) and MCG (cb = 1, legacy)

```
3INST:  x = (s * 89226354 + 64248484) mod 2^32       # QTIP 3.1.1 Algorithm 2 constants
MCG:    x = (s * 0xCBAC1FED) mod 2^32                # codebook_mcg_mult (Q)
both:   y = (x & 0x8FFF8FFF) ^ 0x3B603B60            # lop3.b32 ... 0x8fff8fff, 0x3b603b60, 0x6a
        v = fp16( half(y & 0xFFFF) + half(y >> 16) ) # __hadd of the two fp16 halves
```

LOP3 immediate 0x6a = `(a & b) ^ c`. 0x3B60 is QTIP's fp16 "magic number" m = 0.922. **(derived)**
In each half the sign (bit 15) stays random, exponent bits 14..12 become 011, bits 11..10 stay
random (so the exponent is 2^-3..2^0, magnitude in [0.125, 2)), and the 10 mantissa bits stay
random; the sum of two such halves is bell-shaped.

### 4.3 Output distributions (derived: all 65,536 states evaluated on CPU with the exact ops above)

| codebook | mean | std | min | max | distinct values | kurtosis |
|---|---|---|---|---|---|---|
| mul1 | -0.00193 | 1.00031 | -3.45312 | +3.34766 | 913 | 2.703 |
| 3INST | +0.00021 | **1.24371091** | -3.95703 | +3.97266 | 10,598 | 2.880 |
| MCG | -0.00004 | 1.24411 | -3.94922 | +3.94922 | 10,746 | 2.880 |

The quantizer's global constant `codebook_scale = 1.24371088` (Q) is the standard deviation of the
3INST codebook to 7 digits; weights are first normalized to that RMS and the per-tensor `g_scale`
search (8.3) then absorbs the difference for mul1 (std ~1.0). mul1 has only 913 distinct levels
(sums 0..1020 through fp16 rounding) but 65,536 states; state 0 decodes to the minimum -3.45312.

---

## 5. Bitstream, packing and decode offsets

### 5.1 The ring

For integer K a tile is a ring of `256 K` bits, `S[0 .. 256K)`. Position p (0..255, tensor-core
order) **owns** bits `S[pK .. pK + K)`, and its 16-bit state is the window **ending** at its own bits:

```
state(p) = bits S[(p+1)K - 16 .. (p+1)K - 1] (indices mod 256K), first bit = MSB
```

so the low K bits of state(p) are position p's own bits, and its high 16 - K bits are the tail of
the previous positions' bits (for p = 0 they wrap to the end of the ring: tail-biting). Consecutive
states overlap in 16 - K bits: `state(p+1) = ((state(p) << K) & 0xFFFF) | own(p+1)`.

- Pack (PK `pack_trellis_kernel`): for each position in order, `v &= (1 << K) - 1; k -= K; buf |=
  (v << k)`, flushing 16-bit words MSB-first; 16 threads each pack 16 positions (`i = 16 t`) into K
  words (`j = K t`).
- Word order in memory: the integer packer writes `SWAP16(((uint32_t*) s_packed)[t])`, which (see
  open question Q2) makes each little-endian **uint32** word hold 32 consecutive stream bits with the
  earliest bit at bit 31. The fractional packer states the same layout directly: "Bits are stored
  MSB-first in 32-bit words", `words[pos >> 5] |= 1u << (31 - (pos & 31))` (FR). Canonical form for
  a port: **tile = 8K uint32 words, word w = S[32w .. 32w+31], S[32w] at bit 31.**
- Decode (DQ `dq`): the window start is `b0 = t_offset * bits + bits - 16 + 256 * bits` (the `+256K`
  keeps it non-negative), `i0 = b0 / 32`, words read `ptr[i % (bits * 256 / 32)]`, and the 16 bits are
  cut with `__funnelshift_r(b, a, shift) & 0xffff` from two consecutive uint32 words a, b.

### 5.2 Fractional K (K = KA + 0.5)

Per-position width `D(p) = KA + ((MASK >> (p & 15)) & 1)`; with MASK = 0xAAAA odd positions get
KA + 1 bits (FR `frac_d`). Bits per 16 positions `bpb = 16 KA + popcount(MASK)` (= 16 KA + 8); the
ring has `16 * bpb` bits = `256 (KA + 0.5)`. Position p's window ends at
`S(p) = (p >> 4) * bpb + sum_{j = 0 .. (p & 15)} D(j)` and starts at `S(p) - 16` (mod ring), position p
emits the low D(p) bits of its state MSB-first (FR `frac_s`, `frac_window`). Words per tile
`16 KA + 8` uint16. GEMM tiles: `TILE_U16 = 16 * bits + (half_k ? 8 : 0)` (GI); decode
`dq8_half<KA, cb>` alternates KA- and (KA+1)-bit fields (DQ).

---

## 6. Viterbi (one tile)

### 6.1 Recursion

Nodes between steps are **edges** = the `16 - K`-bit overlaps: `edges = 65536 >> K` (QTK). For
out-edge e (the low 16 - K bits of state(p)) and K-bit candidate k:

```
state   = (k << (16 - K)) | e          # QTK: const int state0 = (k << Kr) | out_edge_idx; Kr = 16 - K
in_edge = state >> K                    # the previous position's out-edge
C_p(e)  = min over k = 0 .. 2^K - 1 of  C_{p-1}(in_edge) + (value(state) - w_p)^2
B_p(e)  = argmin in_edge                # stored: temp_edges[edges * ri + out_edge_idx] = min_in_edge
```

(With the out-edge e fixed, the free K bits at step p are the top, oldest K bits of state(p); this
is the ring of 5.1 parametrized by the overlap after each step. The backtrace rebuilds each state as
`state = (B_p(e) << K) | e`, QTK `encoded = (prev_edge << K) | edge`; the overlapping bits agree.)

- **Targets and arithmetic are fp16** (QTK): `sh_input_tile[i] = __float2half_rn(input_tile[i])`,
  `dh2 = __hsub2(decoded2, w2)`, first step `__hmul2(dh2, dh2)`, later steps
  `__hfma2(dh2, dh2, __half2half2(temp_costs_inc[in_edge_idx]))`; `H_INF = __ushort_as_half(0x7c00)`.
  The QTO kernel also keeps fp16 costs. Out-edges are processed in pairs (e, e+1) that share every
  in-edge (e >> K equal), hence half2.
- **Per-step tie rule:** strict `__hlt` while scanning k = 0, 1, ...: the smallest k wins, which is the
  smallest in_edge (in_edge = `(k << (16 - 2K)) | (e >> K)` is increasing in k). QTO packs
  `(cost_bits << 16) | pred` and takes `min`: the smallest predecessor, and **(derived)** since a
  smaller k is a smaller in-edge, the same tie semantics. Bit-identity of the two kernels is not
  claimed anywhere (open question Q3).
- **Final argmin tie rule** (QTK `argmin_cost`): key = `(half_as_ushort(cost) << 16) | rank`, minimum
  wins, with `v = e & 1023`, `rank = (brev5(v >> 5) << 10) | (brev5(v & 31) << 5) | (e >> 10)` where
  `brev5(x) = __brev(x) >> 27` reverses the low 5 bits; all-infinite -> edge 0.

### 6.2 Tail-biting (QTK / QTF orchestration, QTIP 3.2 Algorithm 4)

```
forward(L / 2, -1);                                   # pass 1: ring rotated by 128, free start
int end_state = backward(L / 2, false, argmin_cost()); # trace back only to position 0
forward(0, end_state);                                # pass 2: position 0's in-edge pinned
backward(0, true, end_state);                         # trace back from out-edge end_state, write
```

`backward` with `write = false` stops after processing ring position 0, so `end_state` is the
in-edge of position 0 (= the overlap between positions 255 and 0) in pass 1's solution. Pass 2 sets
`min_err2 = H_INF` for every candidate whose `in_edge != pre_state` at step 0, and the final
traceback starts from out-edge `end_state` at position 255, which closes the ring exactly. QTIP
Table 2 (L = 12, T = 256): this approximation matches the optimal tail-biting MSE to 4 digits for
k = 1..4.

### 6.3 Outputs

`output_indices[p]` = the 16-bit state of position p (int16 tensor), `output_tile[p] =
__half2float(decode_3inst<cb>(state))`. The host keeps the indices in tensor-core order and packs them
(section 5).

### 6.4 Which kernel runs

QCU `quantize_tiles_use_optimized`: sm_120 always optimized; sm_89 for K in {1,2,4,5,8} or (K = 7
and cb = 0); sm_86 for K in {1,2,4,5,8} or (K = 7 and cb != 2); env `EXL3_QT_OPTIMIZED` overrides.
QTO differs in storage (history as a K-bit choice in uint8, bit-packed for K = 1; pre-decoded values
in registers; K = 6 reads a global decoded table), not in the objective.

### 6.5 CPU check of this section (derived)

A ~100-line numpy check written from this doc (kept outside the repo, in the session scratchpad as
`trellis_ref.py` / `fp16_cost.py` / `cb_stats.py`) implements 4.1-4.2, 5.1 packing and 6.1-6.2 with
float64 costs, on i.i.d. N(0, 1) tiles (32 tiles per K, input scale searched over 0.5..1.5 x
std(codebook)); every solution closed the ring and every pack/unpack round-tripped.

| K | mul1 MSE (exact costs) | 3INST MSE (exact costs) | QTIP Table 2 (L = 12) | Gaussian D(R) = 2^-2K |
|---|---|---|---|---|
| 1 | 0.2714 | 0.2637 | 0.2803 | 0.2500 |
| 2 | 0.0671 | 0.0671 | 0.0733 (Table 1, L = 16: 0.069) | 0.0625 |
| 3 | 0.0174 | 0.0179 | 0.0198 | 0.0156 |
| 4 | 0.00441 | 0.00459 | 0.0055 | 0.0039 |
| 5 | 0.00125 | 0.00123 | -- | 0.00098 |
| 6 | 0.00029 | 0.00032 | -- | 0.00024 |

Best input scale found: 1.1-1.2 x std(cb) at K = 1, 1.0 at K = 2, 0.8-1.0 at K >= 3 (grid step 0.1,
so the K >= 4 rows are a few % pessimistic). Each row uses different random tiles (8,192 samples), so
mul1 vs 3INST differences of a few % are noise: the two codebooks are equivalent in distortion here.
fp16 path costs (EXL3's arithmetic emulated: fp16 targets, `hsub`, `hfma` with one rounding, strict
`<`) vs exact costs on the same 32 tiles (`fp16_cost.py`): MSE +0.02% at K = 2, +0.02% at K = 4,
+0.04% at K = 6. The fp16 costs are not a quality factor; the oracle may use either.

---

## 7. Hessian and block LDLQ

### 7.1 Hessian collection (LIN `capture_H`, CM, CD)

- **What:** for each distinct linear input ("qmap"), `H += x^T x` with `x = input.view(rows,
  in_features).to(torch.float)`, rows containing any non-finite value dropped (`finite =
  torch.isfinite(x).all(dim=1)`), `count += finite rows`. Finalize divides by `count`.
- **Sharing (ARCH):** in Qwen3.5 layers, `q_proj`, `k_proj`, `v_proj` share one H; GDN `in_proj_qkv`
  and `in_proj_z` share one; `gate_proj` and `up_proj` share one; `o_proj`, `out_proj`, `down_proj`
  each have their own. GDN `in_proj_a`, `in_proj_b`, `A_log`, `dt_bias`, `conv1d`, norms stay
  unquantized. This is the same tap set as `hessian_capture.py` (docs/quant2.md 2.1).
- **Sequential:** module by module; after a module is quantized it is reloaded with the quantized
  weights and the calibration state is advanced through it (CM `module.load(..., source =
  q_tensors, ...)`, `advance_state_parallel(...)`), so each H is measured on activations of the
  **already-quantized** preceding model. (r4dx's hessian-v2 is from the bf16 model; see 13.)
- **Calibration** (CD, CM defaults `cal_rows = 250`, `cal_cols = 2048`, 512,000 tokens): rows
  allotted `max(1, int(weight / 135 * rows))` from c4 (20), code (20), multilingual (10), technical
  (10), wiki (50), tiny (5), i.e. 37 + 37 + 18 + 18 + 92 + 9 = 211 text rows, and the last source,
  **uniformly random vocabulary tokens** (`(None, 20, random_data)`, `torch.randint(0, vocab_size,
  (1, columns))` after `torch.manual_seed(0)`), takes the remainder `max(1, rows - len(cal_data))` =
  **39 rows = 15.6%** of the set; BOS/EOS on alternate rows.

### 7.2 Finalize (Q::finalize_capture_H, Q::block_ldl), once per shared H

```
H /= count                                   # count == 0 -> q_fallback
diag_mean = mean(diag H)                     # non-finite or < 1e-20 -> q_fallback
H += sigma_reg * diag_mean * I               # sigma_reg = quant_args.get("sigma_reg", 0.025); never set by CM
diag = diag(H).clone()                       # used by regularize's "auto" out-scale rule only
su_sign = sign(sign(randn(k)) + 1e-5)        # +-1, 0 impossible
H_r = P_k diag(su_sign) H diag(su_sign) P_k  # H *= su.T; had_r(H); H *= su; had_l(H)
L, H = block_ldl(H_r, 16)
L[i, i] = 0 for all i
```

`block_ldl(H, b = 16)`: `L = cholesky(H)` (lower, `H = L L^T`); on failure up to 10 retries, each
adding `2.0 * sigma_reg * mean(diag H)` to the diagonal (cumulative), then re-raise (the first failure
is dumped for debugging; comment: retry damping "has never actually recovered a failing
decomposition"); on OOM, CPU Cholesky. Then `DL = diag blocks of L (m = k/16 blocks of 16 x 16)`,
`L[:, block i] = L[:, block i] @ inv(DL[i])`, diagonal blocks set to I. Result: **H_r = L D L^T**
with L unit block-lower-triangular (16 x 16 identity blocks), `D = blockdiag(DL_i DL_i^T)`. After
`L[i, i] = 0` the diagonal blocks are all zero, so L is the strictly-block-lower feedback matrix.

### 7.3 LDLQ (Q::ldlq), W_r of shape (k, n), rows = input channels

```
buf = 128 (quant_args "buf_size_k", min 16, multiple of 16)
prod_cache = 0 (k x n, fp32)
for j = k, k-128, ..., 128:  i = j - 128                         # spans, LAST rows first
    for bj = 128, 112, ..., 16:  bi = bj - 16                   # 16-row blocks, last first
        comp = prod_cache[i+bi : i+bj] + L[i+bj : j, i+bi : i+bj]^T @ (W_r - Q)[i+bj : j]
        rows = W_r[i+bi : i+bj] + comp                           # 16 x n
        tiles = rows -> (n/16, 256) row-major tiles -> [:, perm]
        q, idx = viterbi(tiles)                                  # section 6, every tile independent
        Q[i+bi : i+bj] = q[:, perm_inv] -> 16 x n;  encoded[(i+bi)/16] = idx
    prod_cache += L[i:j, :]^T @ (W_r - Q)[i:j]                  # lazy update of all earlier rows
```

i.e. `Q[c] = Viterbi(W_r[c] + sum_{r > c, r outside block c} L[r, c]^T E[r])`, `E = W_r - Q`, the
exact block-LDLQ for `tr(E^T H_r E) = || D^(1/2) L^T E ||^2` processed from the last row block to the
first (QTIP Appendix A.2 Algorithm 5 with `T_x = T_y = 16`). Inside a tile the objective is plain
MSE (D's 16 x 16 blocks are ignored), and all n/16 tiles of a row block are independent.

**Hessian basis caveat (exact behaviour, note for the oracle).** L comes from
`P_k diag(su_sign) H diag(su_sign) P_k`, but `regularize` afterwards divides the rows of the weight by
`su = su_sign * r_in / (-1.24371088) / g` (8.2) **before** the input Hadamard. The Hessian that
matches W_r exactly is `P_k diag(su) H diag(su) P_k`; EXL3 does not rebuild L with the magnitudes
(`L` is computed once in finalize and passed straight to `ldlq`; `unrotate_H` uses `.sign()` only).
The oracle MUST reproduce this for fidelity and MAY run a variant with the matching Hessian (open
question Q7). Measured (14 (b)): the matched Hessian lowers the proxy of attention q/k/v by 7-73%
and of the large linears by 0.3-4.4%, at no format or runtime cost.

### 7.4 Shared-Hessian groups (Q::quantize_exl3_batch, CM)

Linears with the same (qmap, K, in_features) are grouped (column cap `max(first.out_features,
min(max_cat_cols, int(3.5e8) // k))`). Each tensor is regularized separately (own sv, own
in-channel scales, own g_scale from one batched `g_scale_search_batch`), then the regularized
weights are concatenated along n and run through one LDLQ pass with the shared L (equivalent to
separate passes, since columns never interact), and refit per tensor. Same-shaped tensors with
different Hessians can go through `ldlq_batched` instead: the same recursion (same backward
16-row order) batched over a stack of tensors with per-tensor L.

### 7.5 Post-quantization refit (Q::refit_scales, on unless `no_refit`; CM never sets it)

In the original basis with `H` = the **damped** Hessian (`H_orig = unrotate_H(H_r, su_sign)`),
`W` the original weight and `Q = W_hat` (section 8.4), 2 rounds of:

```
c_j = (q_j^T H w_j) / (q_j^T H q_j)       (1 where the denominator <= 1e-30)   # per output column
Q *= c (columns);  sv *= c
A = (Q Q^T) o H;  A += 1e-6 * mean(diag A) * I;  b = rowsum(Q o (H W))
r = solve(A, b);  r = 1 where not finite or <= 0                               # per input row
Q *= r (rows);   su *= r
```

The trellis is unchanged; only the stored fp16 scales move. Reported as proxy error
`tr(E^T H E) / tr(W^T H W)` before/after.

### 7.6 Other paths (not default)

- **No usable H** (`q_fallback`): `fallback_quant` Viterbi-quantizes the tiles of W_r with no
  feedback; no refit.
- **Two-sided (YAQA-style) Hessian** `ldlq_2hess`: only when a precomputed output-side Hessian
  `H_out` is supplied via `--hessians` (CM loads it only from those files, `hout = unpack_sym(...) if
  'hout' in f.keys() else None`; online capture fills the input side only); objective
  `tr(E^T H_in E H_out)`, tiles visited along anti-diagonals from the far corner, output H regularized
  with `sigma_reg_out` (default 0.025). Not used for the oracle.

---

## 8. Incoherence processing and scales

### 8.1 Random signs

`su_sign` (k) is drawn in `finalize_capture_H`, `sv_sign` (n) in `quantize_exl3`, both as
`(torch.randn(len).sign() + 1e-5).sign()` after `torch.manual_seed(quant_args["seed"])`, seed = the
module index (CM `"seed": idx`). The signs end up inside `suh`/`svh`, so the exact random stream is
not part of the format.

### 8.2 `regularize` (Q::regularize, exact order)

```
# output side (per column j of W (k, n))
c = rms over k of W[:, j];  c /= mean(c)                       # block_rms(weight, dim = 0), mean > 1e-30
zero = c < 1e-30
if apply_out_scales:                                           # CM default --out_scales always -> True
    c[zero] = 0.1;  sv = sv_sign * c + 1e-10
else:  sv = sv_sign
W = W / sv                                                     # divide columns
sv[zero] = 0                                                   # dead output channels reconstruct to 0
W = W P_n                                                      # blockwise_preapply_had_r_(weight, 128)
# input side (per row i)
r = rms over n of W[i, :];  r[r < 1e-30] = 0.1                 # block_rms(weight, dim = 1)
su = su_sign * r / (-1.24371088) + 1e-10                       # note the minus sign (exact source)
W = W / su                                                     # divide rows: every row now has RMS 1.24371088
W = P_k W                                                      # blockwise_preapply_had_l_(weight, 128)
# global scale
g = g_scale_gss(W)                                             # 8.3
W *= g;  su /= g
return W (= W_r), su, sv
```

(`apply_out_scales = None` ("auto") would enable output scales only when the top 2% of `sqrt(diag
H)` hold < 15% of its sum; the default is "always".) `block_rms` is a plain RMS over a dimension,
computed in 32-wide chunks. Hadamard application: `had_r`: `x.view(k, -1, 128) @ P_128`, `had_l`:
`P_128 @ x.view(-1, 128, n)`, with `P_128 = get_hadamard_dt(128, ..., scale = 1/sqrt(128))`; HPY
builds power-of-two sizes by the Sylvester recursion `[[h, h], [h, -h]]` from `hadamard_1.txt`
(checked before the Paley branch, so 128 is Sylvester even though 127 is a prime = 3 mod 4).

### 8.3 Global scale search (Q::sample_scale_tiles, g_scale_search_batch, g_scale_gss)

- **Samples:** on the (tiles_k x tiles_n) tile grid of W_r, `diag_len = max(tiles_k, tiles_n)`,
  tiles `(i mod tiles_k, (i + w) mod tiles_n)` for i < diag_len, w in {0, 1, 2}, **i-major** (`ii =
  arange(diag_len).repeat_interleave(3)`, `ww = arange(3).repeat(diag_len)`); then
  `num_x = min(max(8, 3 diag_len // 16), (tiles + 1) // 2)` tiles of highest mean square in
  descending order (`torch.topk(tile_ms, num_x)`) and num_x of lowest in ascending order
  (`largest = False`); in tensor-core order. The order matters: stage 1 scores every third sample.
  Multiplied by the **drift** factor `LDLQ_DRIFT = {1: 1.08, 1.5: 1.035,
  2: 1.018, 2.5: 1.009, 3: 1.004}`, 1.0 otherwise (models the growth of values under LDLQ feedback).
- **Objective:** `mean((viterbi(t * s) / s - t)^2)` over the sample tiles (no Hessian).
- **Stage 1:** s in {0.1, 0.3, ..., 1.9} on every third sample tile (`s[::3]`).
- **Stage 2:** 5 points `c + 0.075 (i - 2)` around the stage-1 argmin c, on all samples; if the best
  is interior, parabolic step `0.5 (y0 - y2) / (y0 - 2 y1 + y2)` (only if the denominator > 0),
  clamped to [-0.5, 0.5], times 0.075; `g = max(best + offset, 0.01)`.

### 8.4 Reconstruction and the runtime formula

```
W_hat = diag(su) P_k Q P_n diag(sv)          # Q::quantize_exl3: had_l(Q); *= su; had_r(.); *= sv
y     = x W_hat = ((((x o suh) P_k) Q) P_n) o svh
```

with `suh = fp16(su)`, `svh = fp16(sv)` (k and n values, fp16; packed-sign variants `su`/`sv` as
int16 bitfields also load, EX `unpack_bf`). The 1/sqrt(128) is inside each P.

---

## 9. Per-tensor bit allocation (AL::create_q_strategy)

- Candidate rates: integers 1..8, plus 1.5 / 2.5 / 3.5 when `half_steps` (= codebook is mul1) and
  below 4 (`rate_floor`: `floor(2 bpw)/2 if half_steps and bpw < 4 else floor(bpw)`, clamped to
  1..8; `rate_next`: +0.5 below 4, else +1, capped at 8). So **4.5 is never a per-tensor rate**: a
  4.5 bpw model is a 4/5 mix; 3.5 is uniform 3.5.
- Budget `max_bits = int(bpw * sum_numel)` over linears with `qbits_key == "bits"`; scale overheads
  are not counted (DOC: "Final model bitrate may be somewhat higher than requested").
- Every group starts at `rate_floor(target)`; then repeatedly, over groups sorted by
  `(-max priority, min(layer, last_layer - layer), layer, idx)` (layers nearest either end of the
  stack first), each group whose whole promotion fits is raised one rate step, until nothing fits
  (`cost = sum(t.delta_1() for t in target_list)`, promoted only when `cost > 0 and sum_bits + cost
  <= max_bits`).
- **Groups** are the linears' `qgroup` (`target_groups[module.qgroup]`, idx = the group's first
  appearance in module order, layer = the first `.<digits>.` of the key). For this model: attention
  `q_proj`, `k_proj`, `v_proj` share `key + ".qkv"`, `o_proj` is `key + ".o"` (modules/attn.py); GDN
  `in_proj_qkv` and `in_proj_z` share `key + ".qkvz"`, `out_proj` is `key + ".o"`, `in_proj_a/b` have
  `qmap = None` (not quantized) (modules/gated_delta_net.py); MLP `gate_proj` and `up_proj` share
  `key + ".gu"`, `down_proj` is `key + ".d"` (modules/mlp.py, registered up, gate, down). Attention
  comes before the MLP in a block. **Priorities:** `priority = max(priority, module.q_priority)` from
  0, and none of architecture/qwen3_5.py, attn.py, gated_delta_net.py or mlp.py sets `q_priority`, so
  every group has priority 0.
- lm_head at `head_bits` (default 6), MTP at `mtp_bits` (default 4), outside the budget. `--hq`
  raises `select_hq_bits` layers (2 extra for MoE o/out_proj/shared experts; 0 for this dense model,
  ARCH).

---

## 10. Storage per weight (derived, Qwen3.8-27B shapes)

Per linear: `K k n` trellis bits + `16 (k + n)` bits (suh + svh fp16) + a 4-byte codebook marker; no
per-tile, per-group or zero-point data. Quantized decoder linears (24.327 G params; all k, n
multiples of 128):

| linear (count) | k x n | scale overhead (bpw) |
|---|---|---|
| gdn.in_proj_qkv (48) | 5120 x 10240 | 0.00469 |
| gdn.in_proj_z (48) | 5120 x 6144 | 0.00573 |
| gdn.out_proj (48) | 6144 x 5120 | 0.00573 |
| attn.q_proj incl. gate (16) | 5120 x 12288 | 0.00443 |
| attn.k_proj, v_proj (16 each) | 5120 x 1024 | 0.01875 |
| attn.o_proj (16) | 6144 x 5120 | 0.00573 |
| mlp.gate, mlp.up, mlp.down (64 each) | 5120 x 17408 / 17408 x 5120 | 0.00404 |
| **all decoder linears** | | **0.00447** |

| K | decoder linears bpw | decoder linears GiB | + lm_head at 6 bpw (0.889 GiB) |
|---|---|---|---|
| 2 | 2.0045 | 5.677 | 6.565 |
| 2.5 | 2.5045 | 7.093 | 7.981 |
| 3 | 3.0045 | 8.509 | 9.397 |
| 3.5 | 3.5045 | 9.925 | 10.813 |
| 4 | 4.0045 | 11.341 | 12.229 |
| 4.5 (4/5 mix) | 4.5045 | 12.757 | 13.645 |
| 5 | 5.0045 | 14.173 | 15.061 |
| 6 | 6.0045 | 17.005 | 17.893 |

For comparison: quant2 today 13.68 GiB decode bytes at ~4.5 bpw; UD-Q4_K_XL 15.35 GiB at ~5.05 bpw.

---

## 11. Runtime decode path

### 11.1 Dispatch (EX, GC, GV)

- rows > 144 (`AUTO_RECONSTRUCT_THRESHOLD`): `had_r_128(x, xh, suh)`; reconstruct the dense fp16 weight
  (`ext.reconstruct` or, for rows >= 1024 with k, n % 128 == 0, `reconstruct_had_slice` with both
  Hadamards and signs folded, "ORIGINAL-basis weights"); `hgemm`; `had_r_128(y, y, None, svh)`.
- rows <= 144: `exl3_gemm` (via the C++ `BC_LinearEXL3`), which first tries the QTIP-style GEMV
  (`EXL3_GEMV_MAX_M = 8`; requires suh/svh, k and n % 128 == 0, K in 2..4 (half-K 1.5..3.5), and not
  (K != 4 and cb == 0)), else the tiled GEMM.
- m <= 2 with mul1 and K <= 5 (Ampere/Ada) or <= 6 (Hopper/Blackwell): the **int8-activation GEMV**
  is on by default (`EXL3_INT8_GEMV` unset -> mode 2).

### 11.2 Transforms inside the kernels (GK, GVK, HI)

- Input: a grid-synchronous prologue `had_hf_r_128_inner<true, false>(A, A_had, suh, 0.088388347648f)`:
  x * suh (pre-scale), 128-point Hadamard (each lane holds 4 elements: an in-register 4-point stage,
  then `__shfl_xor` stages 1, 2, 4, 8, 16; fp32 arithmetic), times 1/sqrt(128), rounded to fp16
  into a scratch `A_had`; `grid.sync()`.
- Main loop: trellis tiles streamed with `cp.async` (`TILE_U16 = 16 K (+8)` uint16 per tile),
  decoded to fp16 B fragments (`dq_dispatch`), `mma.m16n8k16` with fp32 accumulators (fp16
  accumulate on sm_86, folded per k-slice); stream-K over the tile grid with an fp32 workspace and
  locks.
- Epilogue: `had_ff_r_128_inner<false, true>` (or `had_hf`) on each 128-column group, post-scaled by
  `svh`.
- GEMV: same prologue/epilogue, per 16 x 16 tile one m16n8k16 pair with **fp16 accumulation folded to
  fp32 every 4 (narrow) / 2 (wide) tiles**; warps split k and never synchronize in the main loop.

### 11.3 Per-weight decode work (from the source; SASS not inspected, open question Q12)

| stage | ops per weight | notes |
|---|---|---|
| state extraction, K = 4 | ~1.1 | per lane of 8 weights: 2 uint32 loads (one shared with the neighbour lane), 1 funnel shift, 7 BFE, 1 AND (DQ `dq8_aligned_4bits`) |
| state extraction, other K | ~1-2 | aligned BFE paths for K = 1, 2; funnel shifts otherwise; half-K `dq8_half` |
| mul1 value | 3 | IMUL, DP4A (the `+0x6400` is the dp4a accumulator), 1/2 PRMT (pack two 16-bit results), 1/2 HFMA2 |
| 3INST value | 3.5 | IMAD, LOP3, then per pair 2 PRMT (lows/highs) + 1 HADD2 |
| MCG value | 3.5 | IMUL instead of IMAD |
| **total, fp16 GEMM/GEMV path (mul1)** | **~4 int/fp ops + the MMA** | QTIP 3.1.1 quotes 3 ALU ops for 3INST, <= 4 for 1MAD, excluding extraction |
| int8 GEMV (mul1) | **~3: extract, IMUL, DP4A** | no fp16 decode, no MMA (below) |

**int8-activation GEMV (G8, G8K).** mul1 is affine in the byte sum, so with int8 activations the
dot product needs no per-weight float work: `w *= 0x83DCD12Du`, activation splat
`((uint8_t)(int8_t) a) * 0x01010101u`, `dp4a.u32.s32(w, splat, acc)` = `a * bytesum(w)`, and per output
`y = k_inv * (q * acc) + (1024 k_inv + k_bias) * (q * sum_a)`. Activations (after the input Hadamard)
are quantized per 16-row k-slice: `scale = max|a| / 127`, `__float2int_rn`, clamp +-127. Mode 1 adds a
second int8 pass on the residual with scale `q / 254` ("~15-16 bit effective activation precision,
KL at parity with fp16 or better"); mode 2 (default) is plain int8 ("~0.9% output RMS deviation").
Note it uses the exact affine value, not the fp16-rounded codebook value the Viterbi optimized
against (tiny mismatch).

**RDNA4 mapping (derived, for planning only).** Every op above has a gfx12 VALU equivalent:
`v_mul_lo_u32` (IMUL; its issue rate on gfx1201 is open question Q13), `v_dot4_u32_u8` /
`v_dot4_i32_iu8` (DP4A, docs/research/r9700-arch.md), `v_alignbit_b32` / `v_bfe_u32` (funnel shift /
BFE), `v_perm_b32` (PRMT), `v_pk_fma_f16` / `v_pk_add_f16`, `v_bfi_b32` / `v_xor3`-class ops for LOP3.
The B-fragment permutation (3.2) must be re-derived for WMMA 16x16x16; the ring format is otherwise
layout-free.

---

## 12. Open questions (not guessed)

- **Q1 line ranges.** Pin every citation to line ranges from a local checkout of `6b84a21` (needs a
  download; not allowed in this workflow).
- **Q2 SWAP16.** Its definition was not seen (not in PK). The reading "swap the two uint16 halves of a
  uint32" is inferred from the pack loop (16-bit words flushed MSB-first), the DQ window reads
  (`funnelshift_r(b, a, s)` over consecutive uint32 words) and FR, which writes MSB-first uint32 words
  directly. Verify against `util.cuh` before emitting EXL3-compatible files (the oracle does not need
  the bytes).
- **Q3 kernel identity.** Whether QTK and QTO produce bit-identical states (same fp16 FMA order?) is
  not stated in the source.
- **Q4 allocation inputs.** *Closed*: the qgroups are q/k/v, in_proj_qkv + in_proj_z, gate + up
  (o_proj, out_proj, down_proj alone) and every priority is 0 (section 9). At the oracle's four points
  the grouping changes nothing (3.5 / 4 / 5 are uniform; the 4.5 budget is used up exactly by whole
  layers 0-15 and 48-63, half of the numel, with zero slack); `allocate` implements it anyway.
- **Q5 calibration rows.** *Closed*: the random-token source takes the remainder, 211 text rows + 39
  random rows at 250 (7.1). The exact bundled corpus files were not inspected.
- **Q6 format doc.** README links `doc/exl3.md`; it is 404 at this commit. No format doc beyond code.
- **Q7 Hessian basis.** Is the sign-only L (7.3 caveat) intentional? *Measured* (14 (b)): on the
  large linears (down, gate, o, in_proj_qkv/z) the matched basis buys 0.3-4.4%, but on attention
  q/k/v it buys 7-73% of the proxy (L3 k_proj 0.00263 -> 0.00070; in the exl3 basis it is worse than
  the int4 baseline), and the refit recovers only part of the gap. Whether it is intentional stays open; for r4dx it
  is a converter-only choice (no format or runtime cost), so an RDNA4 converter should factor the
  matched Hessian, and the oracle is run both ways.
- **Q8 tile_len 160.** `quantize_tiles` also accepts 160-long tiles (mul1 only); purpose not found
  (not used for 16 x 16 linear tiles).
- **Q9 published KL numbers.** exllamav3's KL charts are presumably measured through the long-row
  reconstruct path (weights-only); not confirmed.
- **Q10 batch seeding.** The exact random-draw order of su/sv in `quantize_exl3_batch` vs
  `quantize_exl3` (irrelevant to the format, relevant only to bit-reproducing EXL3 files).
- **Q11 H_out files.** Format of `--hessians` precomputed files (`hout` packed symmetric) not studied.
- **Q12 SASS counts.** 11.3 counts come from source, not disassembly.
- **Q13 RDNA4 op rates.** gfx1201 issue rate of `v_mul_lo_u32` and `v_dot4_u32_u8` for this mix is
  unmeasured (needs a microbench, a GPU run).
- **Q14 GEMM shape table.** `EXL3_GEMM_TILESIZE_K/N`, `BLOCKDIM`, stage counts (a parent header) not
  read; not needed for the oracle.
- **Q15 embeddings.** How EXL3 stores `embed_tokens` for qwen3_5 (believed unquantized fp16) was not
  read. It matters only if the oracle is compared tensor-for-tensor with a GGUF that quantizes it.

---

## 13. Oracle notes for r4dx (what to replicate, what differs)

- **Replicate exactly:** 16 x 16 tiles in the 3.2 order; L = 16 bitshift trellis, tail-biting per
  6.2; mul1 codebook (4.1, fp16 values); regularize 8.2 (both-side signs + 128-blocked Sylvester
  Hadamards + suh/svh + g_scale search with drift); LDLQ 7.3 with 16-row blocks backwards, damping
  0.025 x mean diag, sign-only L; refit 7.5; K per 9 (3.5 = uniform frac, 4.0 and 5.0 uniform, 4.5 a
  4/5 mix by 9's order); lm_head at 6 bpw only if the oracle quantizes lm_head at all (the rung-4
  comparison to UD-Q4_K_XL is weights-only; decide and record which tensors are substituted).
- **Allowed deviations (record them):** fp32/fp64 Viterbi costs instead of fp16 (MSE within 0.04%,
  6.5); EXL3's exact random streams; the NVIDIA permutation.
- **Known differences from EXL3 conversion:** r4dx Hessians (hessian-v2: 410 x 2048 = 834,820
  tokens) are from the **bf16** model (manifest `semantics`: "X = the linear's INPUT activation over
  every calibration token (bf16 checkpoint forward)"), not sequential, with no random-token rows. On
  the unrotated checkpoint the oracle uses the manifest's `keys` (the linear's actual, norm-weighted
  input), not `rms_keys` (weightless-norm input, for folding `(1 + w)` into W as in quant2 Q2a).
- **Calibration domain (a bias, recorded in every manifest's recipe).** hessian-v2's text is
  **domain-matched to the KL corpus**: its `corpus.sources` are WikiText-2 train 262,144 tokens, this
  repo's own code 98,304, text self-generated by the quantized model (chat 167,936, english_prose
  114,688, thai_prose 83,968, code 69,632, multilingual 30,720) and calib.txt + kv_calib_corpus 7,428
  (834,820 in all). The manifest's caveat makes it disjoint from `kl_corpus/` at the text level, not
  by domain: the KL files are cpp_source, english_prose, python_source and thai_prose, so the
  calibration covers exactly those, including this repo's C++/Python style and Thai. EXL3's default
  (7.1) is 512,000 tokens of generic text (c4, code, multilingual, technical, wiki, tiny) of which
  15.6% are uniform random tokens. So the oracle's KL on tokens_canon is **probably optimistic**
  relative to what a stock EXL3 conversion would score there, and the comparison with UD-Q4_K_XL
  (0.0071, quantized with Unsloth's own calibration, not ours) is **not like for like: the bias
  favours the oracle**. The comparison with quant2 (~0.0162) is fair: quant2 uses the same Hessians.
  To predict a stock EXL3 conversion, capture a Hessian set on EXL3's default mix (random rows
  included) and compare the proxy on a few layers; not done here.
  Container names map onto EXL3's per-projection linears: `attn.qg` = q_proj incl. its gate,
  `mlp.gate_up` = gate + up (EXL3 regularizes and scale-searches them separately, then LDLQs them
  together with the shared H). The L7 dead channel (`post_attention_layernorm` ch 3994,
  docs/quant2.md) gives a zero row and column in L7's mlp_in H; the 0.025 damping keeps it positive
  definite, and the matching input row of gate/up only ever multiplies zero.
- **Compute (derived):** a full pass evaluates 2 x 256 x 65,536 candidates per tile = 3.4e7; 24.33 G
  weights = 95.0 M tiles -> 3.2e15 candidate evaluations per rate (+~9% for the scale search), each
  a decode + fp sub + fma + compare. A fused HIP Viterbi (one workgroup per tile, costs in LDS, like
  QTK) is minutes-to-an-hour on one R9700; a non-fused torch implementation is memory-bound at
  roughly a day per rate. Backtrace history is 256 x 2^(16-K) entries per tile (K = 2: 4 MiB as
  uint8 k-choices), so tiles go in waves.

---

## 14. The oracle as built (`tools/reference/trellis_quant.py`)

Section 13, implemented. Everything below is reference tooling: no engine, converter or container
change.

| piece | where | spec |
|---|---|---|
| codebooks (mul1 / 3inst / mcg), exact fp16 semantics | `codebook_np` | 4 |
| tile order, rate widths, words per tile | `tensor_core_perm`, `widths`, `frac_k`, `trellis_words` | 3 |
| reference Viterbi (any state width / length / K, two-pass or pinned) | `viterbi_torch` | 6 |
| native encoders, bit-identical to the reference | `trellis_viterbi.hip` (CPU build: std::thread; HIP build: gfx1201, one kernel per rate), `TrellisEncoder` | 6 |
| stored bitstream: 8K uint32 words per tile, MSB-first | `pack_states`, `unpack_states`, `decode_words` | 5 |
| regularize (signs, 128-point Hadamards, suh/svh), global scale search | `regularize`, `g_scale_search`, `sample_scale_tiles` | 8.2, 8.3 |
| damping, sign-rotated H, block LDL, block LDLQ | `damp_hessian`, `rotate_hessian`, `block_ldl`, `ldlq` | 7.2, 7.3 |
| un-rotate, refit, reconstruct from the stored bits | `unrotate`, `refit_scales`, `reconstruct` | 7.5, 8.4 |
| one shared-H group end to end (equal-K tensors concatenated along n, as EXL3) | `quantize_group` | 2, 7.4 |
| allocation (EXL3's qgroups, 4.5 = 4/5 mix, ends of the stack first) | `allocate`, `mix` | 9 |
| our int4 g64 LDLQ baseline (quant_ldlq.hpp in torch) | `int4_ldlq` | -- |

**Hessians.** hessian-v2's `keys` files (the linear's real, post-norm input from the bf16 forward),
as section 13 says for the unrotated oracle. The L7 `mlp_in` dead channel (zero row and column) is
kept positive definite by the 0.025 damping; `tests/reference/test_trellis_quant.py` runs a group with
a dead input channel.

**Recorded deviations** (in every manifest's `recipe`): fp32 path costs, smallest-edge final argmin,
our own seeded sign draws, bf16-forward non-sequential Hessians, calibration domain-matched to the KL
corpus with no random-token rows (13: probably flatters the oracle against a stock EXL3 conversion
and against UD-Q4_K_XL, not against quant2), lm_head and embeddings kept bf16 (the oracle quantizes
the 24.33 G decoder-linear weights only). Not deviations: the allocation (EXL3's qgroups, every
priority 0; section 9) and the global-scale sample order (EXL3's, 8.3). Q7 is reproduced
(`--hessian-basis exl3`, the default: the EXL3-fidelity oracle); `--hessian-basis matched` factors
`P diag(su) H diag(su) P` per tensor instead (the Hessian of the weight LDLQ actually quantizes), the
"what would our converter ship" oracle (b).

**Output.** `quantize-model --K <rate> --out-dir <dir>` writes one `L<ii>.safetensors` per layer
(`<hf name>.trellis` int32 words `[k/16, n/16, 8K]`, `.suh` fp16 `[k]`, `.svh` fp16 `[n]`), a
per-layer JSON record and `weights_override.json` (per-tensor K, measured bits, proxy losses before
and after refit, relative weight error, g scale, seeds, timings, the layer file's sha256; a summary
per class). It resumes layer by layer: a layer is kept only when its record's job (checkpoint,
Hessians, basis, recipe AND the sha256 of trellis_quant.py and trellis_viterbi.hip) matches and every
tensor has this run's K; anything else is redone, and the manifest lists such records as
`stale_layers` and never counts them towards `complete`. `mix --bpw 4.5 --src <K4 dir> --src <K5
dir>` writes a manifest that takes each tensor from the directory of the K the allocator gives it (a
tensor's trellis depends only on the tensor, its H, K and seeds), so 4.5 costs no quantization; it
refuses sources of another checkpoint, Hessian set or basis, and records (and warns about) sources
written by different code. `full_logits_golden.py
--weights-override <dir>` reconstructs `W_hat = diag(suh) P_k decode(words) P_n diag(svh)` on the
device and checks every tensor against the bf16 weight and against the error recorded when its bits
were written.

**Validation** (2026-09-26, this tree; the HIP encoder on device 1 is bit-identical to the CPU one
and to the torch reference: `tests/reference/test_trellis_quant.py` OK with HIP, `selftest --device
cuda --tiles 256` identical states and costs at K = 3, 3.5, 4, 4.5, 5, 6):

- (a) iid N(0, 1), 2048 tiles per K, best of a 0.02-step scale grid (0.70-1.30 x std(cb)), mul1, CPU
  encoder: MSE 0.2728 / 0.0685 / 0.0173 / 0.00881 (3.5) / 0.00444 / 0.00225 (4.5) / 0.00115 /
  0.00030 at K = 1, 2, 3, 3.5, 4, 4.5, 5, 6; 3INST 0.0689 / 0.00463 at K = 2 / 4. QTIP Table 1
  (L = 16, 2 bits, 1MAD/3INST): 0.069; Table 2 (L = 12): 0.0733 / 0.0198 / 0.0055 at K = 2..4.
  Scalar Lloyd-Max at the same rate: 0.3634 / 0.1175 / 0.0345 / 0.0095 / 0.0025 / 0.00065 (K = 1..6),
  so the trellis is 1.3x (K = 1) and 1.7-2.2x (K = 2..6) lower. Every ring closed, every pack/unpack
  round-tripped, and the measured bits per weight equal K exactly.
- (b) the Hessian proxy `tr(E^T H E) / tr(W^T H W)` (undamped hessian-v2 H, from the stored bits,
  after the refit; the quantize-model seeds), trellis K = 4 (4.004-4.019 bpw incl. suh/svh) in both
  Hessian bases against our converter's w4a16 g64 LDLQ (4.5 bpw; `int4_ldlq`, damp 0.01, group
  search on the updated weights) with and without a 128-block random Hadamard on the input side (a
  stand-in for q2ab's online Hadamards). HIP encoder, `linear --device cuda`:

  | linear (k x n) | trellis K=4, exl3 basis | matched basis | matched vs exl3 | int4 g64 LDLQ | + input RHT128 |
  |---|---|---|---|---|---|
  | L10 mlp.down (17408 x 5120) | 0.002191 | 0.002184 | -0.3% | 0.003971 | 0.003618 |
  | L7 mlp.gate (5120 x 17408, dead channel) | 0.001877 | 0.001868 | -0.4% | 0.003244 | 0.003177 |
  | L11 attn.o (6144 x 5120) | 0.000353 | 0.000350 | -0.7% | 0.000695 | 0.000576 |
  | L10 gdn.in_proj_z (5120 x 6144) | 0.000411 | 0.000399 | -3.1% | 0.000799 | 0.000665 |
  | L10 gdn.in_proj_qkv (5120 x 10240) | 0.000847 | 0.000809 | -4.4% | 0.001672 | 0.001359 |
  | L63 attn.v (5120 x 1024) | 0.000327 | 0.000304 | -7.2% | 0.000752 | 0.000511 |
  | L31 attn.k (5120 x 1024) | 0.000779 | 0.000669 | -14.1% | 0.001552 | 0.001099 |
  | L3 attn.q incl. gate (5120 x 12288) | 0.000060 | 0.000043 | -27.7% | 0.000094 | 0.000066 |
  | L3 attn.v (5120 x 1024) | 0.001992 | 0.001275 | -36.0% | 0.003083 | 0.002097 |
  | L3 attn.k (5120 x 1024) | **0.002634** | 0.000700 | **-73.4%** | 0.002218 | 0.001127 |

  At 0.5 bpw less the trellis is below the int4 baseline on every linear but one: 35-56% in the
  exl3 basis, 42-68% in the matched one; **L3 attn.k in the exl3 basis is 19% worse than int4** (and
  2.3x worse than int4 + RHT). **The Hessian basis (Q7) matters on attention q/k/v.** On the large
  linears (down, gate, o, in_proj_qkv/z) the matched basis buys only 0.3-4.4%; on q/k/v it buys
  7-73%. The exl3-basis L is built from `P S_u H S_u P` with signs only, but LDLQ quantizes a weight
  whose rows were divided by the input-scale magnitudes `r / 1.2437 / g` (8.2), so its feedback
  minimizes a different metric wherever those magnitudes matter; matched is exact on the input side
  (the output scales still weight columns, which neither basis models). The spread of the
  magnitudes is a plausible driver but not the whole story (|suh| / mean, K = 4 smoke run: L3 k_proj
  0.27-2.04, cv 0.10; gate/up cv 0.024-0.036; but L3 o_proj and L0 gdn.out_proj spread about as much,
  cv 0.08-0.11, and o_proj gained 0.7% at L11); how it interacts with H's correlations was not
  studied. The refit recovers only part of the gap (L3 k: 0.00325 -> 0.00263 in exl3; matched needs
  no refit, 0.000702 -> 0.000700), and the exl3 result there is fragile: a 2e-6 relative change in g
  (the sample order fix, 8.3) moved L3 k by 2% (0.002687 -> 0.002634) and v by 2% while matched
  moved by 0.02%, and quantize-model's concatenated q/k/v LDLQ pass (same math, other fp32 GEMM
  blocking on the GPU) gives L3 k 0.00270, v 0.00201. **Consequences:** the basis is a
  converter-only choice with no format or runtime cost, so an RDNA4 trellis converter should factor
  the matched Hessian, and the first lever for k/v is the basis, not extra bits. The default oracle
  (`--hessian-basis exl3`) measures EXL3 as it is and so understates what our own converter would
  reach; judge the <= 0.008 gate on the matched oracle (`--hessian-basis matched`), with the exl3
  one alongside for fidelity.
- (c) the encoders, `bench` (16k-tile batches; HIP also 64k): CPU (9950X, 32 threads) 7,500 /
  11,000 / 8,300 / 9,600 / 12,700 tiles/s at K = 3.5 / 4 / 4.5 / 5 / 6 (~6,800 inside
  quantize-model, whose batches are one 16-row block), 2.3-3.9 h of encoder time per rate for the
  95.03 M tiles (+9% scale search). **HIP (R9700, device 1, measured): 26,900 / 29,100 / 29,600 /
  30,700 / 30,600 tiles/s** (64k tiles; 24,800-30,900 at 16k), i.e. **0.94-1.07 h of encoder time per
  rate**. Inside `linear --device cuda` a whole linear including regularize, LDL, LDLQ, refit and
  the reconstruction check takes 23 s (down_proj), 15 s (gate), 9-11 s (in_proj_qkv, o), 1-7 s
  (k, v); `quantize-model --K 4 --layers 3` (one attention layer, 7 linears) took 70 s at 28,600
  tiles/s, so a full rate is about 1.3-1.5 h on device 1 in either basis. `quantize-model` refuses
  to start if the HIP encoder differs from the CPU one in a single state.

### 14.1 Full-model results (the A0 gate)

Run on 2026-09-26 with `tools/quant2/trellis_oracle.ps1` on device 1, code be15724. Each rate
quantizes all 400 decoder linears; the K=4 run took 73 min at about 28,000 tiles/s, and 4.5 is
`mix` over K4m and K5m. Scoring used the same weights-only
reference forward (`full_logits_golden.py --weights-override`) with the same tokens
(`tokens_canon.json`, 4 segments, canonical Thai) and the same bf16 reference
(`D:\models\r4dx\kl-canon\ref`) as the UD-Q4_K_XL row. Every run checked 400/400 tensors, and each
point's measured bpw equals its target plus 0.0045 for suh/svh. Results are in
`D:\models\r4dx\kl-trellis\summary.json`.

| model | bpw (decoder linears) | decoder GiB | mean KL | top-1 | p99 KL | cpp | en | py | thai |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| trellis K3.5m | 3.5045 | 9.93 | 0.01611 | 94.18% | 0.1165 | 0.0136 | 0.0174 | 0.0145 | 0.0190 |
| **trellis K4m** | 4.0045 | 11.34 | **0.00813** | 95.94% | 0.0544 | 0.0066 | 0.0082 | 0.0077 | 0.0100 |
| trellis mix4.25m | 4.2545 | 12.05 | 0.00688 | 96.53% | | 0.0055 | 0.0076 | 0.0064 | 0.0081 |
| **trellis mix4.5m** | 4.5045 | 12.76 | **0.00547** | 96.90% | 0.0387 | 0.0044 | 0.0063 | 0.0050 | 0.0063 |
| UD-Q4_K_XL (GGUF, weights only) | ~5.05 all text weights | 15.35 decode | 0.00706 | 96.38% | | 0.0058 | 0.0082 | 0.0054 | 0.0089 |
| q2ab_hv2_q3 (runtime, today's best) | ~4.5 | 13.68 decode | 0.01555 | 93.65% | | | | | |

`m` = `--hessian-basis matched`. The exl3-basis runs (EXL3 fidelity) were not run at full model; the
per-linear table in (b) predicts they sit slightly above matched, mostly through attention k/v.

**A0 passes.** K4m is at the ~0.008 gate, and mix4.5m is at 0.68 of it. Each half bit roughly
halves the KL (0.0161 → 0.0081 → 0.0055), and the ratio holds in every segment, Thai included.
mix4.25m (EXL3's allocator puts K = 5 on layers 0-7 and 56-63: 100 of the 400 linears) sits
between them at 0.00688, already below UD-Q4_K_XL.

Two cautions when comparing:

- **Versus q2ab_hv2_q3 (fair on Hessians, not on scope).** Both use hessian-v2. But q2ab's 0.01555
  is a runtime number. It includes a 4-bit g32 lm_head, the fp8 KV cache (about +0.0012 on its own)
  and the kernels' activation rounding. The oracle's rows are weights-only, with a bf16 lm_head.
  Gate A2 in `docs/trellis-kernel.md` measures that difference on the real runtime.
- **Versus UD-Q4_K_XL (optimistic).** Both rows are weights-only on the same reference path. But
  the GGUF also quantizes lm_head and embeddings, and hessian-v2 is domain-matched to the KL corpus
  (13). So the oracle is flattered, by an amount not measured here. Even so, mix4.5m's 0.00547 at
  12.76 GiB of decoder weights against 0.00706 at 15.35 GiB of decode bytes leaves a wide margin.

---

## 15. References

- exllamav3, commit `6b84a21b6f1e5da3f291b9e1019061f0de788279`,
  https://github.com/turboderp-org/exllamav3 (MIT). Files as tabled in 1.1.
- A. Tseng, Q. Sun, D. Hou, C. De Sa, "QTIP: Quantization with Trellises and Incoherence
  Processing", arXiv 2406.11235: 2.1 (incoherence processing, RHT `W <- V_m S_m W S_n V_n^T`), 3.1
  (bitshift trellis), 3.1.1 (1MAD, 3INST, Algorithms 1-2), 3.2 (tail-biting, Algorithm 4), Table 1
  (2-bit Gaussian MSE: 1MAD/3INST 0.069, D(R) 0.063), Table 2 (tail-biting approximation), Appendix
  A.2 (BlockLDLQ with trellis, Algorithm 5).
- In-tree: `docs/quant2.md` (LDLQ, rotation, rms Hessians, dead channel), `tools/reference/
  hessian_capture.py`, `src/convert/include/r4dx_convert/{quant_ldlq,dense_linalg,hessian_store}.hpp`.
