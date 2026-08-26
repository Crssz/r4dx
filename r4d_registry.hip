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
// `op` is the key callers select on, and it is the one field several rows may share: fp8 and bf16
// KV are two kernels for one op, and so are the exact and the lossy all-reduce. Rows are in
// preference order within an op -- select() returns the first that matches.
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
static const R4DConstraint cAttnPrefillBf16[] = {
    C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 16), C_EQ("causal", 1),
    C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
};
// q_len <= 10 is the q_len*gqa <= 64 of the prose, at the gqa 6 this kernel is compiled for.
static const R4DConstraint cAttnDecodeFp8[] = {
    C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 16), C_EQ("causal", 1),
    C_LE("q_len", 10), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3"),
};
static const R4DConstraint cAttnDecodeBf16[] = {
    C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 16), C_EQ("causal", 1),
    C_LE("q_len", 10), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
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

// ---- all-reduce --------------------------------------------------------------------------------
// `exact` separates the two, and a caller has to say which it wants: a request that does not
// declare it will accept a lossy wire format gets no kernel rather than the quantised one.
static const R4DConstraint cArExact[] = {
    C_EQ("world_size", 2), C_EQ("exact", 1), C_IN("dtype", "bf16 fp16 fp32"),
};
static const R4DConstraint cArWht6[] = {
    C_EQ("world_size", 2), C_EQ("exact", 0), C_IN("dtype", "bf16 fp16"), C_DIV("numel", 64),
};

// ---- GEMM --------------------------------------------------------------------------------------
// DFlash2's grouped dynamic depthwise convolution. block_size is not listed as a predicate: the
// body masks the position with block_size-1, so only powers of two are admissible, and that is a
// property of the value rather than a bound select() can express. The entry point rejects it.
static const R4DConstraint cDflashConv[] = {
    C_EQ("taps", 2), C_EQ("group_size", 16), C_DIV("hidden_size", 128),
    C_IN("dtype", "bf16"),
};

static const R4DConstraint cGemmNt[] = {
    C_LE("M", 16), C_DIV("K", 256), C_IN("dtype", "bf16"),
};

static const R4DConstraint cGemmNt64[] = {
    C_LE("M", 64), C_DIV("K", 16), C_IN("dtype", "bf16"),
};
static const R4DConstraint cGemmNtW4[] = {
    C_LE("M", 64), C_DIV("K", 128), C_DIV("N", 16), C_IN("dtype", "w4a16"),
};
static const R4DConstraint cGemmNtW4A8[] = {
    C_LE("M", 64), C_DIV("K", 128), C_DIV("N", 16), C_IN("dtype", "w4a8"),
};
static const R4DConstraint cGemmNtMxfp4A8[] = {
    C_LE("M", 64), C_DIV("K", 32), C_DIV("N", 16), C_IN("dtype", "mxfp4a8"),
};
static const R4DConstraint cQuantActI8[] = {
    C_DIV("K", 16), C_IN("dtype", "bf16"),
};

#define ROW(arr) arr, (int)(sizeof(arr) / sizeof(arr[0]))

static const R4DKernelInfo kKernels[] = {
    {"attn_prefill_h256_gqa6_fp8kv", "attn", "attn_prefill_paged",
     "paged causal attention, query-tiled (prefill / chunked prefill)",
     "head_dim 256, 6 queries per kv head, paged block 16, varlen",
     "bf16 query, fp8-e4m3 kv cache with per-(seq,head) descales, bf16 out",
     ROW(cAttnPrefillFp8)},
    {"attn_prefill_h256_gqa6_bf16kv", "attn", "attn_prefill_paged",
     "paged causal attention, query-tiled (prefill / chunked prefill)",
     "head_dim 256, 6 queries per kv head, paged block 16, varlen",
     "bf16 query, bf16 kv cache, bf16 out",
     ROW(cAttnPrefillBf16)},
    {"attn_decode_h256_gqa6_fp8kv", "attn", "attn_decode_paged",
     "paged causal attention, split-KV (decode / speculative verify)",
     "head_dim 256, 6 queries per kv head, paged block 16, q_len*gqa <= 64",
     "bf16 query, fp8-e4m3 kv cache with per-(seq,head) descales, bf16 out, f16 partials",
     ROW(cAttnDecodeFp8)},
    {"attn_decode_h256_gqa6_bf16kv", "attn", "attn_decode_paged",
     "paged causal attention, split-KV (decode / speculative verify)",
     "head_dim 256, 6 queries per kv head, paged block 16, q_len*gqa <= 64",
     "bf16 query, bf16 kv cache, bf16 out, f16 partials",
     ROW(cAttnDecodeBf16)},
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
    {"ar_oneshot_2rank_exact", "ar", "allreduce",
     "all-reduce (sum), one-shot push over P2P IPC scratch",
     "exactly 2 ranks, peer-accessible; scratch sized by the full message; cudagraph-safe",
     "bf16 / fp16 / fp32 payload, fp32 accumulate, bit-identical across ranks",
     ROW(cArExact)},
    {"ar_oneshot_2rank_wht6", "ar", "allreduce",
     "all-reduce (sum) with a Walsh-Hadamard-rotated 6-bit wire payload -- lossy",
     "exactly 2 ranks, peer-accessible; groups of 64 elements; cudagraph-safe",
     "bf16 / fp16 payload, 6 bits + bf16 scale per group on the wire, fp32 accumulate",
     ROW(cArWht6)},
    {"gemm_bf16_nt_m16", "gemm", "gemm_nt",
     "skinny GEMM C[M,N] = A[M,K] @ W[N,K]^T, split-K reduced in LDS",
     "M <= 16; K divisible by SK and K/SK by 256; N, K otherwise runtime",
     "bf16 A / W / C, fp32 accumulate, deterministic reduction",
     ROW(cGemmNt)},
    {"gemm_bf16_nt_m64", "gemm", "gemm_nt",
     "skinny GEMM C[M,N] = A[M,K] @ W[N,K]^T with 16x16x16 WMMA, split-K reduced in LDS",
     "M <= 64; K divisible by SK*16; N, K otherwise runtime",
     "bf16 A / W / C, fp32 accumulate, deterministic reduction",
     ROW(cGemmNt64)},
    {"gemm_w4a16_nt_m64", "gemm", "gemm_nt",
     "skinny GEMM C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T with a 4-bit weight, 16x16x16 WMMA, "
     "split-K reduced in LDS",
     "M <= 64; N a multiple of 16; K divisible by SK*128; weight pre-permuted into fragment order",
     "f16 A, 4-bit W with an f16 scale and integer zero per 128 K, bf16 C, fp32 accumulate",
     ROW(cGemmNtW4)},
    {"gemm_w4a8_nt_m64", "gemm", "gemm_nt",
     "skinny GEMM C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T with a 4-bit weight and an 8-bit "
     "activation, 16x16x16 int8 WMMA, split-K reduced in LDS",
     "M <= 64; N a multiple of 16; K divisible by SK*128; weight pre-permuted into fragment order "
     "and SHARED byte for byte with gemm_w4a16_nt_m64; activation byte-shuffled inside each k step",
     "int8 A with an f32 per-row scale, signed 4-bit W with an f16 scale per 128 K, bf16 C, "
     "int32 accumulate per group",
     ROW(cGemmNtW4A8)},
    {"gemm_mxfp4a8_nt_m64", "gemm", "gemm_nt",
     "skinny GEMM C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T with an OCP-MXFP4 weight and an fp8 "
     "activation, 16x16x16 fp8 WMMA, split-K reduced in LDS",
     "M <= 64; N a multiple of 16; K divisible by SK*32; weight pre-permuted into fragment "
     "order (32 lanes x uint32 per n-tile/k-step), one E8M0 exponent per 32 K in [K/32][N] "
     "order, plus a per-row reference exponent",
     "e4m3 A with an f32 per-row scale, e2m1 W with an E8M0 scale per 32 K folded into the "
     "element against a per-row reference, bf16 C, fp32 accumulate",
     ROW(cGemmNtMxfp4A8)},
    {"quant_act_i8", "quant", "quant_act",
     "per-row symmetric int8 quantisation of a bf16 activation, written in the byte order "
     "gemm_w4a8_nt_m64's A fragment reads",
     "K a multiple of 16; one workgroup per row",
     "bf16 in, int8 out with an f32 per-row scale",
     ROW(cQuantActI8)},
    {"dflash_conv_t2_g16_bf16", "dflash", "dflash_conv",
     "DFlash2 grouped dynamic depthwise conv: coefficient add, both taps and the block-position "
     "mask in one pass",
     "taps 2, group_size 16, hidden_size divisible by group_size*8, block_size a power of two; "
     "delta is a slice of the [T, 2, taps, NG] projection so its row pitch is 2*taps*NG; out must "
     "not alias x",
     "bf16 x / delta / base / out, fp32 accumulate",
     ROW(cDflashConv)},
};

extern "C" int r4d_kernel_count(void) {
    return (int)(sizeof(kKernels) / sizeof(kKernels[0]));
}

extern "C" const R4DKernelInfo* r4d_kernel_at(int i) {
    if (i < 0 || i >= r4d_kernel_count()) return nullptr;
    return &kKernels[i];
}
