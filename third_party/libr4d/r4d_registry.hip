// r4d_registry.hip: what this build of the library actually contains.
//
// R4D kernels are specialised, and their names say so -- but a name is only useful to a human
// reading the source. This table is the machine-readable form: a caller that wants to know whether
// R4D covers the model it is about to serve reads it, instead of hardcoding a copy of the
// constraints and drifting from them. Adding a kernel means adding a row here; the build fails to
// mean anything otherwise, since this is also what `r4d.kernels()` prints.
//
// Each row states its geometry twice. `shape` is the prose, and lists every constraint the entry
// point rejects on -- including the ones that are NOT in the name because they are the same across
// the whole library (paged block size 16, bf16 query, gfx1201). `constraints` is the same geometry
// as predicates, which is what `r4d.select()` matches a caller against. The predicates are
// deliberately a NECESSARY condition only: a rule that cannot be written as one predicate on one
// named parameter -- the GEMM's K/SK relationship, for instance -- stays in the prose and in the
// entry point, and select() answers the weaker question. Where the two disagree the prose is the
// one to trust and the row is a bug.
//
// `op` is the key callers select on, and it is the one field several rows may share (the GEMM
// families). Rows are in preference order within an op -- select() returns the first that matches.
#include "r4d.h"

#define C_EQ(k, v)  {k, R4D_C_EQ, (v), nullptr}
#define C_LE(k, v)  {k, R4D_C_LE, (v), nullptr}
#define C_GE(k, v)  {k, R4D_C_GE, (v), nullptr}
#define C_DIV(k, v) {k, R4D_C_DIV, (v), nullptr}
#define C_IN(k, s)  {k, R4D_C_IN, 0, s}

// ---- attention ---------------------------------------------------------------------------------
// `causal` is a constraint, not a note: the paged kernels are causal by construction and the vision
// kernel is dense, and neither can serve the other's masking.
static const R4DConstraint cAttnPrefillFp8[] = {
    C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 16), C_EQ("causal", 1),
    C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3"),
};
// q_len <= 10 is the q_len*gqa <= 64 of the prose, at the gqa 6 this kernel is compiled for.
static const R4DConstraint cAttnDecodeFp8[] = {
    C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 16), C_EQ("causal", 1),
    C_LE("q_len", 10), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3"),
};
// The sliding-window family (r4d_attn_paged_h256_gqa2.hip): the same kernels at gqa 2, plus a window.
// `window` >= 1 is a constraint a caller selects with; q_len <= 32 is the q_len*gqa <= 64 of the prose.
static const R4DConstraint cAttnPrefillWinFp8[] = {
    C_EQ("head_dim", 256), C_EQ("gqa", 2), C_EQ("block_size", 16), C_EQ("causal", 1),
    C_GE("window", 1), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3"),
};
static const R4DConstraint cAttnDecodeWinFp8[] = {
    C_EQ("head_dim", 256), C_EQ("gqa", 2), C_EQ("block_size", 16), C_EQ("causal", 1),
    C_GE("window", 1), C_LE("q_len", 32), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3"),
};
static const R4DConstraint cAttnVit[] = {
    C_EQ("head_dim", 72), C_EQ("gqa", 1), C_EQ("causal", 0),
    C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
};

// ---- gated delta net ---------------------------------------------------------------------------
static const R4DConstraint cGdnChunkScan[] = {
    C_EQ("head_k", 128), C_EQ("head_v", 128), C_EQ("chunk", 64),
};
static const R4DConstraint cGdnConvPrep[] = {
    C_EQ("conv_width", 4), C_EQ("head_k", 128), C_EQ("head_v", 128), C_EQ("chunk", 64),
};
static const R4DConstraint cGdnConvUpdate[] = {
    C_EQ("conv_width", 4), C_EQ("head_k", 128), C_EQ("head_v", 128),
};
static const R4DConstraint cGdnKktSolve[] = {
    C_EQ("head_k", 128), C_EQ("chunk", 64),
};
static const R4DConstraint cGdnRecurrentUpdate[] = {
    C_EQ("head_k", 128), C_EQ("head_v", 128),
};
static const R4DConstraint cGdnGatedRmsNorm[] = {
    C_EQ("channels", 128),
};

// ---- GEMM --------------------------------------------------------------------------------------
// DFlash2's grouped dynamic depthwise convolution. block_size is not listed as a predicate: the
// body masks the position with block_size-1, so only powers of two are admissible, and that is a
// property of the value rather than a bound select() can express. The entry point rejects it.
static const R4DConstraint cDflashConv[] = {
    C_EQ("taps", 2), C_EQ("group_size", 16), C_DIV("hidden_size", 128),
    C_IN("dtype", "bf16"),
};

static const R4DConstraint cGemmNt64[] = {
    C_LE("M", 64), C_DIV("K", 16), C_IN("dtype", "bf16"),
};
static const R4DConstraint cGemmNtW4[] = {
    C_LE("M", 64), C_DIV("K", 128), C_DIV("N", 16), C_IN("dtype", "w4a16"),
};
// The trellis GEMM's full entry (output transform included) is the gemm_nt row; the raw one stops
// short of the linear, so it gets its own op and a caller selecting "gemm_nt" is never handed it.
static const R4DConstraint cGemmNtTrellis[] = {
    C_LE("M", 64), C_DIV("K", 128), C_DIV("N", 128),
    C_IN("dtype", "trellis_mul1_k4 trellis_mul1_k5"),
};
static const R4DConstraint cGemmNtTrellisRaw[] = {
    C_LE("M", 64), C_DIV("K", 128), C_DIV("N", 128),
    C_IN("dtype", "trellis_mul1_k4 trellis_mul1_k5"),
};
// The M = 128 / 256 sibling is its own op: a caller selecting "gemm_nt" (M <= 64) is never handed it.
static const R4DConstraint cGemmNtTrellisM256[] = {
    C_DIV("M", 128), C_LE("M", 256), C_DIV("K", 128), C_DIV("N", 128),
    C_IN("dtype", "trellis_mul1_k4 trellis_mul1_k5"),
};
// The int8 x int8 sibling (r4d_gemm_trellis_nt_i8): exactly 256 rows, its own op, so "gemm_nt_m256" callers are
// never handed a quantized kernel.
static const R4DConstraint cGemmNtTrellisI8[] = {
    C_EQ("M", 256), C_DIV("K", 128), C_DIV("N", 128),
    C_IN("dtype", "trellis_mul1_k4 trellis_mul1_k5"),
};
static const R4DConstraint cTrellisReconstruct[] = {
    C_DIV("K", 16), C_DIV("N", 32), C_IN("dtype", "trellis_mul1_k4 trellis_mul1_k5"),
};

#define ROW(arr) arr, (int)(sizeof(arr) / sizeof(arr[0]))

static const R4DKernelInfo kKernels[] = {
    {"attn_prefill_h256_gqa6_fp8kv", "attn", "attn_prefill_paged",
     "paged causal attention, query-tiled (prefill / chunked prefill)",
     "head_dim 256, 6 queries per kv head, paged block 16, varlen",
     "bf16 query, fp8-e4m3 kv cache with per-(seq,head) descales, bf16 out",
     ROW(cAttnPrefillFp8)},
    // Its own op: it needs a scratch buffer and a caller-chosen split count, so a caller selecting
    // plain "attn_prefill_paged" is never handed it.
    {"attn_prefill_splitkv_h256_gqa6_fp8kv", "attn", "attn_prefill_paged_splitkv",
     "paged causal attention, query-tiled and split-KV (long-context prefill), fixed-order merge",
     "head_dim 256, 6 queries per kv head, paged block 16, varlen, caller's split count <= 64",
     "bf16 query, fp8-e4m3 kv cache with per-(seq,head) descales, bf16 out, fp32 partials",
     ROW(cAttnPrefillFp8)},
    // Its own op: same bits as attn_prefill_h256_gqa6_fp8kv, more workgroups (a->splits = geometry).
    {"attn_prefill_exact_h256_gqa6_fp8kv", "attn", "attn_prefill_paged_exact",
     "paged causal attention, query-tiled, PV columns split across workgroups (bit-identical to plain)",
     "head_dim 256, 6 queries per kv head, paged block 16, varlen",
     "bf16 query, fp8-e4m3 kv cache with per-(seq,head) descales, bf16 out",
     ROW(cAttnPrefillFp8)},
    {"attn_decode_h256_gqa6_fp8kv", "attn", "attn_decode_paged",
     "paged causal attention, split-KV (decode / speculative verify)",
     "head_dim 256, 6 queries per kv head, paged block 16, q_len*gqa <= 64",
     "bf16 query, fp8-e4m3 kv cache with per-(seq,head) descales, bf16 out, f16 partials",
     ROW(cAttnDecodeFp8)},
    // Own ops: they take R4DArgsW (a window, and klimit_ext on prefill), so a caller selecting plain
    // "attn_prefill_paged" / "attn_decode_paged" is never handed them.
    {"attn_prefill_h256_gqa2_fp8kv", "attn", "attn_prefill_paged_window",
     "paged causal sliding-window attention, query-tiled (prefill / chunked prefill), optional "
     "per-row bidirectional key limit (klimit_ext)",
     "head_dim 256, 2 queries per kv head, paged block 16, varlen, window W >= 1; a sliding KV ring "
     "is a block table whose entries repeat; the cache must be zero-initialised",
     "bf16 query, fp8-e4m3 kv cache with per-(seq,head) descales, bf16 out",
     ROW(cAttnPrefillWinFp8)},
    {"attn_decode_h256_gqa2_fp8kv", "attn", "attn_decode_paged_window",
     "paged causal sliding-window attention, split-KV over the window's tiles (decode / speculative "
     "verify)",
     "head_dim 256, 2 queries per kv head, paged block 16, q_len*gqa <= 64, window W >= 1, scratch "
     "required; the split law sees min(max_ctx, W + q_len + 16)",
     "bf16 query, fp8-e4m3 kv cache with per-(seq,head) descales, bf16 out, f16 partials",
     ROW(cAttnDecodeWinFp8)},
    {"attn_vit_h72_bf16", "attn", "attn_vit",
     "dense non-causal vision-encoder attention, varlen over images / windows",
     "head_dim 72 native (no padding to 128), multi-head (no GQA), contiguous q/k/v/o, "
     "one cu_seqlens segment per image",
     "bf16 query / key / value / out, f16 operands, fp32 accumulate",
     ROW(cAttnVit)},
    {"gdn_chunk_scan_k128_v128_c64_bf16", "gdn", "gdn_chunk_scan",
     "gated delta net chunked scan: WY recompute, state recurrence and output in one kernel",
     "head_k 128, head_v 128, chunk 64, varlen, state resident in WMMA accumulators",
     "bf16 q/k/v/A, fp32 gate/beta/state, bf16 out, fp32 final state",
     ROW(cGdnChunkScan)},
    {"gdn_conv_prep_w4_h128_bf16", "gdn", "gdn_conv_prep",
     "gdn prefill preamble: causal conv + qkv split + qk l2norm + gating + per-chunk cumsum",
     "conv width 4, head_k/head_v 128, chunk 64, varlen, conv state read and rewritten in place",
     "bf16 x / weights / conv state / q,k,v out, fp32 a,b and gate/beta out",
     ROW(cGdnConvPrep)},
    {"gdn_conv_update_w4_h128_bf16", "gdn", "gdn_conv_update",
     "gdn decode conv: rolling speculative window over the conv state cache, writes q/k/v",
     "conv width 4, head_k/head_v 128, state cache width-1 + num_spec, read at accepted offset",
     "bf16 x / weights / conv state / q,k,v out",
     ROW(cGdnConvUpdate)},
    {"gdn_kkt_solve_k128_c64_bf16", "gdn", "gdn_kkt_solve",
     "gdn chunk preamble: A = (I + strict_lower(diag(beta) K K^T e^dg))^-1, gram never in HBM",
     "head_k 128, chunk 64, varlen; one gram per k head, applied to every v head of the group",
     "bf16 k and A out, fp32 beta / gate / gram / inverse",
     ROW(cGdnKktSolve)},
    {"gdn_recurrent_update_k128_v128_bf16_fp32state", "gdn", "gdn_recurrent_update",
     "gdn decode recurrent update: gating, qk l2norm, delta-rule state update, gated rms norm "
     "and output",
     "head_k 128, head_v 128, paged state, one state written per candidate token",
     "bf16 q/k/v and out, fp32 a,b and state cache",
     ROW(cGdnRecurrentUpdate)},
    {"gdn_gated_rmsnorm_h128_bf16", "gdn", "gdn_gated_rmsnorm",
     "gated rms norm over a row: out = rms(x) . w . act(z), for the prefill path only",
     "128 channels per row, one wave per row; silu or sigmoid gate",
     "bf16 x / z / out, fp32 weight",
     ROW(cGdnGatedRmsNorm)},
    {"gemm_bf16_nt_m64", "gemm", "gemm_nt",
     "skinny GEMM C[M,N] = A[M,K] @ W[N,K]^T with 16x16x16 WMMA, split-K reduced in LDS",
     "M <= 64; K divisible by SK*16; N, K otherwise runtime",
     "bf16 A / W / C, fp32 accumulate, deterministic reduction",
     ROW(cGemmNt64)},
    {"gemm_w4a16_nt_m64", "gemm", "gemm_nt",
     "skinny GEMM C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T with a 4-bit weight, 16x16x16 WMMA, "
     "split-K reduced in LDS",
     "M <= 64; N a multiple of 16; K divisible by SK*max(group,64); weight pre-permuted into "
     "fragment order; group = the build default (gemm_w4a16_nt_m64_group), or 32/64 per call "
     "through gemm_w4a16_nt_m64_g",
     "f16 A, 4-bit W with an f16 scale and integer zero per group of K, bf16 C, fp32 accumulate",
     ROW(cGemmNtW4)},
    {"gemm_trellis_nt_m64", "gemm", "gemm_nt",
     "skinny GEMM with a trellis-coded weight (EXL3/QTIP mul1): C = bf16((FWHT128(A @ Q) * svh) "
     "* out_scale), the whole trellis linear given an input-transformed A; 16x16x16 f16 WMMA, "
     "split-K in LDS and across blocks, the 128-column output transform in the epilogue (the "
     "last-arriving block finishes a split group), one kernel for M <= 64",
     "M <= 64; K, N multiples of 128; tiles in the n-pair grid; block width WV*NP*32 in "
     "{32,64,128,256} dividing N; (K/16) divisible by SK*SKG*U; KB 4 or 5 "
     "(gemm_trellis_nt_m64_has_rate); up to two A parts split at a 128-column boundary",
     "f16 A (x * suh, FWHT-128, prescaled), mul1 trellis W at KB bits per weight, fp32 svh, bf16 "
     "C, fp32 accumulate and transform, one rounding, deterministic reduction",
     ROW(cGemmNtTrellis)},
    {"gemm_trellis_nt_m64_raw", "gemm", "gemm_nt_raw",
     "skinny GEMM C[M,N] = A[M,K] @ Q[K,N] with a trellis-coded weight Q (EXL3/QTIP mul1), "
     "16x16x16 f16 WMMA, split-K in LDS and across blocks (last-arriving block sums a split "
     "128-column group), NO output transform -- the test and bench entry",
     "M <= 64; K, N multiples of 128; tiles in the n-pair grid; block width WV*NP*32 in "
     "{32,64,128,256} dividing N; (K/16) divisible by SK*SKG*U; KB 4 or 5 "
     "(gemm_trellis_nt_m64_has_rate)",
     "f16 A (already input-transformed), mul1 trellis W at KB bits per weight, fp32 C, fp32 "
     "accumulate, deterministic reduction",
     ROW(cGemmNtTrellisRaw)},
    {"gemm_trellis_nt_m256", "gemm", "gemm_nt_m256",
     "the trellis linear of gemm_trellis_nt_m64 for M = 256 (or 128) rows in one launch: the decoded "
     "weights are staged once in LDS for RG = M / 64 row groups; every output element gets, bit for "
     "bit, the bits the M <= 64 kernel gives a 64-row slice at the same (SK, SKG) (K-slice sum in "
     "slice order from 0.f, SKG partials in y order, then the stock FWHT / svh / out_scale / one bf16 "
     "rounding), so it replaces 4 (or 2) M = 64 launches without changing a bit",
     "M 256 (or 128); K, N multiples of 128; (K/16) divisible by SK*SKG, SK in {2,4,8,16} the M = 64 "
     "row's slice count, at least U = 4 k-tiles per slice; block width 32 columns (ws always used: "
     "M*N*SKG fp32); only the instantiated (KB, skw, SK/skw, k-tile tail) combinations "
     "(gemm_trellis_nt_m256_check); up to two A parts split at a 128-column boundary",
     "f16 A (x * suh, FWHT-128, prescaled), mul1 trellis W at KB bits per weight, fp32 svh, bf16 "
     "C, fp32 accumulate and transform, one rounding, deterministic reduction, <= 190 VGPRs and no "
     "scratch in every instantiation (build-checked)",
     ROW(cGemmNtTrellisM256)},
    {"gemm_trellis_nt_i8", "gemm", "gemm_nt_i8",
     "the trellis linear of gemm_trellis_nt_m256 for exactly 256 rows with QUANTIZED operands: A as int8 per "
     "(row, 128 k) (r4d_trellis_i8_quant_act), the decoded weight as int8 per (column, 128 k) on the fly from a "
     "scale table (r4d_trellis_i8_wscale), v_wmma_i32_16x16x16_iu8, the int32 partial sum of every 128 k rescaled "
     "into fp32, then the f16 kernel's FWHT / svh / out_scale / one bf16 rounding. NOT bit-identical to "
     "gemm_trellis_nt_m256",
     "M 256; K, N, n_split multiples of 128; (K/128) divisible by skw*skg, skw in {2,4,8}, skg in {1,2,4,8}; "
     "ws always used (M*N*SKG fp32); up to two A parts split at a 128-column boundary",
     "int8 A8 + fp32 SA, mul1 trellis W at KB bits per weight + fp32 weight-scale table, fp32 svh, bf16 C, int32 "
     "per-128-k sums, fp32 rescale and transform, one rounding, deterministic reduction, <= 192 VGPRs and no "
     "scratch in every instantiation (build-checked)",
     ROW(cGemmNtTrellisI8)},
    {"trellis_reconstruct_f16", "quant", "trellis_reconstruct",
     "the whole decoded trellis weight Q[K,N] as f16, with the GEMM's own device decode",
     "K a multiple of 16, N of 32 (tile pairs); KB 4 or 5",
     "mul1 trellis W in the n-pair grid in, f16 Q out, bit-exact against the reference decode",
     ROW(cTrellisReconstruct)},
    {"dflash_conv_t2_g16_bf16", "dflash", "dflash_conv",
     "DFlash2 grouped dynamic depthwise conv: coefficient add, both taps and the block-position "
     "mask in one pass",
     "taps 2, group_size 16, hidden_size divisible by group_size*8, block_size a power of two; "
     "delta is a slice of the [T, 2, taps, NG] projection so its row pitch is 2*taps*NG; out must "
     "not alias x",
     "bf16 x / delta / base / out, fp32 accumulate",
     ROW(cDflashConv)},
};

// The attention geometries this build serves (r4d.h's R4DAttnGeom), in the order the units are listed
// in third_party/CMakeLists.txt. The numbers restate each unit's own #defines (A_* in
// r4d_attn_paged_h256_gqa6.hip, W_* in r4d_attn_paged_h256_gqa2.hip).
static const R4DAttnGeom kAttnGeoms[] = {
    {"h256_gqa6", 256, 6, 16, 64, 0},
    {"h256_gqa2", 256, 2, 16, 64, 1},
};

extern "C" int r4d_attn_geom_count(void) {
    return (int)(sizeof(kAttnGeoms) / sizeof(kAttnGeoms[0]));
}

extern "C" const R4DAttnGeom* r4d_attn_geom_at(int i) {
    if (i < 0 || i >= r4d_attn_geom_count()) return nullptr;
    return &kAttnGeoms[i];
}

extern "C" int r4d_kernel_count(void) {
    return (int)(sizeof(kKernels) / sizeof(kKernels[0]));
}

extern "C" const R4DKernelInfo* r4d_kernel_at(int i) {
    if (i < 0 || i >= r4d_kernel_count()) return nullptr;
    return &kKernels[i];
}
