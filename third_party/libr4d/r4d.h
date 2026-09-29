// R4D -- the RDNA4 (gfx1201) kernel library: attention, gated delta net and skinny GEMMs (bf16,
// w4a16, trellis), compiled into one static library in r4dx. Plain C ABI: every entry point takes raw device
// pointers and a stream, so the driver can be a pybind module or ctypes with no torch/C++ ABI
// coupling. Nothing on the launch paths allocates or synchronises, so any of them can be recorded
// into a HIP graph.
//
// NAMING. An entry point is
//
//     r4d_<family>_<op>_<geometry it is compiled for>
//
// and the geometry suffix is not decoration: these are specialised kernels, and every dimension in
// the name is a compile-time constant that the entry point REJECTS a mismatch on rather than
// running. `attn_decode_h256_gqa6_fp8kv` runs for head_dim 256 with 6 queries per KV head and an
// fp8-e4m3 paged cache, and for nothing else. A model with a different head size needs a new
// instantiation, which will sit beside this one under its own name; nothing has to be renamed to
// make room for it. Dimensions that are fixed for the whole library, and so discriminate nothing
// between entry points, stay out of the names and are reported by r4d_*_dims() and the kernel
// registry instead: the paged block size (16), the query dtype (bf16 everywhere -- see
// r4d_attn_paged_h256_gqa6.hip) and the target architecture.
//
// SOURCE FILES CARRY THE SAME NAME. A translation unit is named for the entry point it provides,
// geometry included -- r4d_gdn_kkt_solve_k128_c64_bf16.hip holds r4d_gdn_kkt_solve_k128_c64_bf16
// and nothing else. Where one unit provides a family that differs only in a suffix, the file name
// stops at the shared part: r4d_gdn_conv_w4_h128_bf16.hip provides the prep and update pair, and
// r4d_attn_paged_h256_gqa6.hip provides all four paged variants by including the two kernel
// templates that sit beside it under their own names. Only the shared machinery -- r4d_common.h,
// r4d_dt16.h, r4d_gdn_wmma.h -- is named for what it is rather than for a kernel, because it
// provides none.
//
// Where a shape is unsupported the attention and GDN entry points return a negative code; the
// GEMMs throw std::runtime_error.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

// Library version. The module exposes it as r4d.__version__, and the git tag it was built from is
// expected to match -- which is what lets a consumer assert it linked the sources it pinned rather
// than whatever a stale clone happened to hold.
#define R4D_VERSION "0.5.0"

struct R4DArgs {
    const void*  q;             // (num_seqs*q_len, q_heads, head_dim)  bf16
    const void*  kv;            // (num_blocks, kv_heads, block_size, 2*head_dim)  fp8 e4m3
                                //   strides below are in ELEMENTS of that dtype; K then V per slot
    const int*   block_table;   // (num_seqs, max_blocks)            int32
    const int*   seqused_k;     // (num_seqs,)                       int32
    void*        out;           // (total_q, q_heads, head_dim)      bf16
    const float* k_descale;     // (num_seqs, kv_heads)
    const float* v_descale;     // (num_seqs, kv_heads)
    const float* q_descale;     // unused: the query is bf16
    void*        scratch;       // split-KV partials (decode, split-KV prefill), or null
    int num_seqs, q_len, q_heads, kv_heads, head_dim, block_size, max_blocks;
    int64_t kv_block_stride;       // elements between consecutive blocks
    int64_t kv_head_stride;        // elements between kv heads inside a block
    float scale;
    int  splits;                // decode: 0 = let the split law choose; split-KV prefill: the
                                //   segment count (<= 1 = unsplit); ignored by plain prefill
    int  max_ctx;               // host-visible context bound (seqused_k is device-side)
};

extern "C" {
// ---- attention: paged, causal, varlen ------------------------------------------------------
// Compiled for head_dim 256, 6 queries per KV head, paged block size 16, bf16 query. The prefill
// kernel tiles the query; the decode kernel splits the KV and takes at most 64 query rows
// (q_len * gqa), the band a speculative-decode verify step falls in.
// Return 0 on success, negative on a shape this instantiation does not serve.
int  r4d_attn_prefill_h256_gqa6_fp8kv (const R4DArgs* a, hipStream_t stream);
int  r4d_attn_decode_h256_gqa6_fp8kv  (const R4DArgs* a, hipStream_t stream);
// Bytes of split-KV partial buffer one decode launch of this shape needs (f16 partials).
int64_t r4d_attn_decode_h256_gqa6_scratch_bytes(const R4DArgs* a);
// Split-KV prefill: the prefill kernel with the KV tile range cut into a->splits segments (the
// caller's count, clamped to 64; there is no split law), each writing fp32 partials to a->scratch,
// then the decode path's fixed-order merge. For long contexts, where the plain launch has only
// ceil(q_len/64) x kv_heads workgroups. splits <= 1 IS the plain prefill launch (same kernel, same
// bits, no scratch). Split, the result differs from the unsplit launch only in where each segment's
// softmax reference max starts and in the fp32 order of the merge. -5 if split and scratch is null.
int  r4d_attn_prefill_splitkv_h256_gqa6_fp8kv (const R4DArgs* a, hipStream_t stream);
// Bytes of fp32 partials one split-KV prefill launch of this shape needs (0 when unsplit).
int64_t r4d_attn_prefill_splitkv_h256_gqa6_scratch_bytes(const R4DArgs* a);
// Exact-wide prefill (fp8 KV): the plain prefill's output bit for bit, over more workgroups (fewer
// warps per q-block and/or the PV columns split across DS workgroups that each redo QK/softmax).
// a->splits: 0 = default geometry, 1 = the plain launch, NW*10+DS = a measured geometry, else -3.
int  r4d_attn_prefill_exact_h256_gqa6_fp8kv(const R4DArgs* a, hipStream_t stream);
// The geometry the attention kernels above are compiled for, so a caller can test a model against
// it instead of discovering the mismatch at the first launch.
void r4d_attn_dims(int* head_dim, int* gqa, int* block_size, int* max_decode_rows);

// ---- attention: vision encoder -------------------------------------------------------------
// Dense, non-causal, multi-head attention at the vision tower's head_dim of 72, bf16 throughout.
// Nothing is paged and nothing is quantised: q, k, v and o are [total_tokens, heads, 72] and
// contiguous, and cu_seqlens [num_seqs + 1] bounds one image (or one attention window) per entry,
// so a whole batch is ONE launch rather than a per-segment loop and a concatenate. `max_seqlen` is
// the host-side bound on those lengths and selects the query-block height; the kernel reads the
// device-side cu_seqlens itself. head_dim 72 is carried natively (5 k-tiles of 16, the last half
// structurally zero) rather than padded to 128. Returns -1 for any other head size.
int  r4d_attn_vit_h72_bf16(const void* q, const void* k, const void* v, void* o,
                           const void* cu_seqlens, int num_seqs, int max_seqlen,
                           int heads, int head_dim, float scale, void* stream);
// head_dim, the two query-block heights, and the segment length at which the launcher switches.
void r4d_attn_vit_dims(int* head_dim, int* rows_large, int* rows_small, int* split);

// ---- gated delta net -----------------------------------------------------------------------
// One chunked scan over N variable-length sequences: the WY recompute, the recurrent state scan and
// the output in one kernel (FLA's recompute_w_u_fwd + chunk_gated_delta_rule_fwd_h + chunk_fwd_o).
// Layouts, all contiguous and bf16 unless stated:
//   q,k [T, Hg, K]   v,o [T, H, V]   A [T, H, bt]   g,beta [T, H] fp32
//   h0,ht [N, H, V, K] fp32          cu [N+1] int32
// K, V and bt are the compile-time 128/128/64 in the name; a mismatch returns -1 rather than
// running. Named for the algorithm, not the serving phase: this is the chunked scan, which is what
// prefill and chunked prefill both run. Decode uses the recurrent update, a different kernel.
int r4d_gdn_chunk_scan_k128_v128_c64_bf16(
        const void* q, const void* k, const void* v, const void* A,
        const void* g, const void* beta, const void* h0, void* o, void* ht,
        const void* cu, int N, int H, int Hg, int K, int V, int bt,
        float scale, void* stream);
void r4d_gdn_dims(int* head_k, int* head_v, int* chunk);

// The chunk preamble: A = (I + strict_lower(diag(beta) K K^T e^{g_i-g_j}))^-1 per chunk, which is
// FLA's chunk_scaled_dot_kkt_fwd followed by solve_tril with the fp32 gram in between never
// reaching HBM. k [T,Hg,K] bf16; beta, g [T,H] fp32 (g already summed along the chunk);
// A [T,H,64] bf16 out; cu [N+1] int32. Returns -1 for a geometry it was not compiled for.
int r4d_gdn_kkt_solve_k128_c64_bf16(const void* k, const void* beta, const void* g, void* A,
                                    const void* cu, int N, int T, int H, int Hg, int K, int bt,
                                    void* stream);

// DFlash2's grouped dynamic depthwise convolution, fused into one pass. The reference builds it as
// ~6 full [T, H] elementwise passes plus a materialised [T, taps, num_groups, group] coefficient
// tensor, all consumed exactly once. delta is a SLICE of the [T, 2, taps, NG] kernel projection, so
// dpitch is 2*taps*NG; out must not alias x. block_size must be a power of two.
int r4d_dflash_conv_t2_g16_bf16(
        const void* x, const void* delta, const void* base, void* out,
        int T, int H, int dpitch, int NG, int taps, int group, int block_size, void* stream);

// Everything between the qkv projection and the chunked scan, in one kernel: the depthwise causal
// convolution (width 4, silu) with its state cache, the q/k/v split, the l2 norm on q and k, the
// gate g = -exp(A_log).softplus(a + dt_bias) with its per-chunk cumsum, and beta = sigmoid(b).
// Replaces causal_conv1d_fn + fused_post_conv_prep + chunk_local_cumsum; the conv output never
// reaches HBM. Strides are in ELEMENTS; x may be a padded view (the qkvz split the layer hands it).
int r4d_gdn_conv_prep_w4_h128_bf16(
        const void* x, int64_t xpitch, const void* wgt, const void* bias, void* cstate,
        int64_t cs_seq, int64_t cs_dim, int64_t cs_tok, const void* cache_idx, int64_t ci_stride,
        const void* has_init, const void* a, const void* b, int64_t ab_stride, int ab_is_bf16,
        const void* A_log, const void* dt_bias, void* q, void* k, void* v, void* g, void* beta,
        const void* cu, int N, int T, int H, int Hg, int K, int V, int width, float softplus_thr,
        void* stream);

// The same convolution for a decode step: the tokens are a speculative window, the state cache is
// a rolling buffer of width-1 + num_spec entries read at the slot the last ACCEPTED token left,
// and q / k / v are written straight into their own layouts. Replaces causal_conv1d_update and
// the cat that made its output contiguous.
int r4d_gdn_conv_update_w4_h128_bf16(
        const void* x, int64_t xpitch, const void* wgt, const void* bias, void* cstate,
        int64_t cs_seq, int64_t cs_dim, int64_t cs_tok, int state_len_max, const void* cache_idx,
        int64_t ci_stride, const void* num_accepted, void* q, void* k, void* v, const void* cu,
        int N, int H, int Hg, int K, int V, int width, int max_query_len, void* stream);

// The gated RMS norm the layer applies to its own output: out = rms(x) . w . act(z), one row per
// (token, head). Replaces FLA's rmsnorm_fn. act: 0 = silu/swish, 1 = sigmoid. Only the prefill
// path needs it -- the decode kernel below folds the same arithmetic into its epilogue, because
// its workgroup owns the whole row.
int r4d_gdn_gated_rmsnorm_h128_bf16(const void* x, const void* z, const void* w, void* o,
                                    int64_t rows, int64_t xrow, int64_t zrow, int64_t orow, int width,
                                    float eps, int act, void* stream);

// The recurrent delta-rule update decode runs where prefill runs the chunked scan: gating, the qk
// l2 norm, the state update and the output, against the paged state cache. The state is fp32
// (this model's config asks for it) and one is written per candidate token, which is the whole
// cost of the kernel. Replaces fused_sigmoid_gating_delta_rule_update.
int r4d_gdn_recurrent_update_k128_v128_bf16_fp32state(
        const void* q, const void* k, const void* v, const void* a, const void* b,
        int64_t ab_stride, int ab_is_bf16, const void* A_log, const void* dt_bias, void* state,
        int64_t state_slot_stride, int64_t state_head_stride, void* o, const void* cu,
        const void* ssm_state_indices, int64_t indices_stride, const void* num_accepted,
        const void* z_gate, const void* norm_weight, float norm_eps, int norm_act,
        int N, int H, int Hg, int K, int V, float scale, float softplus_thr, void* stream);

// ---- GEMM ----------------------------------------------------------------------------------
// Skinny bf16 GEMM for M up to 64: C[M,N] = A[M,K] @ W[N,K]^T, bf16 throughout -- a torch Linear
// with the weight stored (N,K), which is the "nt" in the name. Where the weight read dominates and
// rocBLAS does poorly. Computed with 16x16x16 WMMA so a 16-wide step of K costs ceil(M/16)
// activation fragments and one weight fragment, instead of one weight load and M activation loads.
// WV columns per block x SK k-splits, reduced in LDS.
void r4d_gemm_bf16_nt_m64(int64_t a, int64_t w, int64_t c, int M, int K, int N, int WV, int SK, int MB,
                          int64_t stream);
int  r4d_gemm_bf16_nt_m64_max_m(void);

// Skinny GEMM with a 4-BIT weight: C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T, f16 A, bf16 C. The weight
// is asymmetric per output channel per group of `group()` contiguous K -- w ~= scale * (q - zero),
// q in 0..15, zero an integer -- and is pre-permuted offline into the WMMA fragment order, so the
// kernel's whole weight path is one global_load_b128 per lane per four k steps. Wq is N*K/2 bytes;
// Wsz is one dword per (row, group), the f16 scale in its low half and the f16 of -(1024 + zero)
// in its high half. N must be a multiple of 16 and K divisible by SK * group().
// group() is the BUILD DEFAULT (R4D_GEMM_W4_GROUP), which is what r4d_gemm_w4a16_nt_m64 serves.
void r4d_gemm_w4a16_nt_m64(int64_t a, int64_t wq, int64_t wsz, int64_t c, int M, int K, int N,
                           int WV, int SK, int MB, int NPW, int NT, int64_t stream);
int  r4d_gemm_w4a16_nt_m64_max_m(void);
int  r4d_gemm_w4a16_nt_m64_group(void);
// The same kernel at a group chosen per call. has_group() is 1 for 32 and 64, 0 otherwise --
// a build default outside those two is served by r4d_gemm_w4a16_nt_m64 only. The packed Wq is
// the same bytes at every group; Wsz has K/group dwords per row. K must be divisible by
// SK * max(group, 64) -- a split has to start on a 64-K packed block -- and an uninstantiated group
// throws, like any other rejected shape.
int  r4d_gemm_w4a16_nt_m64_has_group(int group);
void r4d_gemm_w4a16_nt_m64_g(int group, int64_t a, int64_t wq, int64_t wsz, int64_t c, int M, int K,
                             int N, int WV, int SK, int MB, int NPW, int NT, int64_t stream);

// Skinny GEMM with a TRELLIS-coded weight: EXL3 / QTIP tiles of 16 k x 16 n, each a tail-biting
// ring of 256*KB bits whose 16-bit states hash to f16 values (the "mul1" codebook), f16 A, 16x16x16
// WMMA, one kernel for M = 1..64, KB = 4 or 5 bits per weight. `w` is the tiles' uint32 ring words
// in the PAIR GRID -- word w of tile (tn, tk) at (((tn >> 1) * (K/16) + tk) * 2 + (tn & 1)) * 8KB + w
// -- N*K*KB/8 bytes, 16-byte aligned. Q[K,N] below is the decoded weight in its regularized domain;
// the linear it belongs to is W^T = diag(suh) H Q H diag(svh) (natural-order 128-point Hadamards,
// each with its 1/sqrt(128)), whose input side -- A = f16(H(x * suh) * 2^s / sqrt(128)) -- is applied
// to A before this call (r4dx's r4dx_trellis_input_bf16).
//
// The linear: C[m][n] = bf16( (FWHT128(A @ Q)[m][n] * svh[n]) * out_scale ), bf16 C [M][N], svh fp32
// [N], out_scale = 2^-s / sqrt(128) (s = the input side's prescale). fp32 from the accumulator through
// the FWHT (stages in natural order, a + b / a - b, FwhtLds's order) and svh, one bf16 rounding (to
// nearest even). Two A parts: output columns >= n_split read a1 (a fused gate/up pair with different
// input transforms), all others a0; a1 = 0 means a0; every 128-column group lies inside one part.
// Tuning: WV x SK waves per block (column blocks x in-block K splits), NP tile pairs per wave (block
// width Wc = 32 WV NP, one of 32/64/128/256), SKG blocks splitting K across the grid, MT row tiles
// per block, U k-tiles per step, NT for non-temporal weight loads. Legal only when K and N are
// multiples of 128, N and n_split of Wc, (K/16) of SK*SKG*U, SK*Wc*32 <= 64 KiB, WV*SK*32 <= 1024,
// and (KB, NP, U, MT) is instantiated (NP, U in {1,2,4}, NP*U <= 8; MT up to 4 at NP 1 and at
// (NP, U) = (2, 1), 3 at (2, 2), 3 at KB 4 / 2 at KB 5 at (2, 4), 2 at (4, 1), 1 at (4, 2) -- what
// fits 190 VGPRs); anything else throws. A 128-column group whose sums come from more than one
// block (SKG > 1 or Wc < 128) is finished (summed, transformed, rounded) by the last block to
// arrive, through `ws` (fp32, ws_bytes(M, N, SKG)) and `tickets` (u32 [N/128], tickets_bytes(N);
// zero before the first call, and every complete launch leaves them zero again -- zero_tickets
// resets a buffer a launch that did not complete left behind), and needs every row tile in one
// block (ceil(M/16) <= MT). One ticket buffer per linear (never shared by two linears in flight).
// A row's result depends on the tuning's summation order (SK, SKG, Wc), never on M or on the other
// rows, so a caller that gives every M <= 16 the same tuning gets verify rows equal to decode rows.
void r4d_gemm_trellis_nt_m64(int64_t a0, int64_t a1, int n_split, int64_t w, int64_t svh,
                             int64_t c, int64_t ws, int64_t tickets, int M, int K, int N, int KB,
                             int WV, int SK, int MT, int NP, int SKG, int U, int NT, float out_scale,
                             int64_t stream);
// _raw is the test and diagnostic entry: fp32 C[M,N] = A @ Q with no output transform (the same
// kernel, reduction and legality), so one-hot A rows return rows of Q bit for bit. `clk` (0 = off):
// 4 uint64 that block (0,0,0)'s thread 0 fills with clock64() / wall_clock64() at entry and at the
// end of its own work, for the shader clock during a real launch.
void r4d_gemm_trellis_nt_m64_raw(int64_t a0, int64_t a1, int n_split, int64_t w, int64_t c,
                                 int64_t ws, int64_t tickets, int M, int K, int N, int KB, int WV,
                                 int SK, int MT, int NP, int SKG, int U, int NT, int64_t clk,
                                 int64_t stream);
int    r4d_gemm_trellis_nt_m64_has_rate(int KB);   // 1 for an instantiated KB (4, 5), else 0
int    r4d_gemm_trellis_nt_m64_max_m(void);         // 64
size_t r4d_gemm_trellis_nt_m64_ws_bytes(int M, int N, int SKG);   // SKG * M * N * 4
size_t r4d_gemm_trellis_nt_m64_tickets_bytes(int N);              // (N / 128) * 4
// hipMemsetAsync(tickets, 0, bytes) on `stream`: the reset after a launch that did not complete.
void   r4d_gemm_trellis_nt_m64_zero_tickets(int64_t tickets, size_t bytes, int64_t stream);
// The whole decoded Q[K,N] as f16 bits (row-major, K rows), from the same pair-grid words and the
// same device decode as the GEMM, bit-exact against the format's reference decode. KB 4 or 5; K a
// multiple of 16, N of 32.
void   r4d_trellis_reconstruct_f16(int64_t w, int64_t q, int K, int N, int KB, int64_t stream);

// M = 256 (or 128) trellis GEMM (r4d_gemm_trellis_nt_m256.hip; docs/trellis-m256.md): the trellis
// linear of r4d_gemm_trellis_nt_m64 for 64 * RG rows in ONE launch, RG = M / 64 = 4 (M = 256) or 2
// (M = 128), with the decoded weights staged once in LDS for every row group. It is NOT a new tuning
// of the M <= 64 kernel: every output element gets, bit for bit, the bits the M <= 64 kernel gives it
// for the M = 64 tuning row with the same (SK, SKG) -- the K-slice sum is a left fold from 0.f in
// slice order, the SKG partials are summed in y order from 0.f, then the stock FWHT stages, svh,
// out_scale and one bf16 rounding -- so a caller that runs 64-row chunks through its shipped M = 64
// row may replace four of them (or two) by one call without changing a bit. Nothing else about the
// M <= 64 kernel, its rows or the M <= 16 decode / verify rows changes.
//   a0 / a1 / n_split / w / svh / c / tickets / out_scale: as r4d_gemm_trellis_nt_m64, with M rows.
//   ws: fp32, r4d_gemm_trellis_nt_m256_ws_bytes(M, N, SKG) -- ALWAYS required (a 32-column block is
//       always a split 128-group), every slot written before it is read.
//   SK: the K-slice count of the M = 64 row to reproduce (2, 4, 8 or 16); SKG likewise (1, 2, 4, 8);
//       NP = tile pairs per wave and U = k-tiles per step, (RG, NP, U) = (4, 1, 4) for M = 256,
//       (2, 2, 1) or (2, 1, 2) for M = 128; skw = K slices resident per workgroup (0 = min(SK, 4)),
//       SK / skw groups of them are walked in turn. Only the (KB, RG, NP, U, skw, SK / skw, k-tile
//       tail) combinations that are instantiated are legal (the build checks each one's 190-VGPR /
//       zero-scratch fit): r4d_gemm_trellis_nt_m256_check says which, without launching.
//   Legal only when K and N are multiples of 128, N of 32 NP, n_split of 128, (K/16) of SK * SKG, a
//   slice has at least U k-tiles, and the instantiation exists; anything else throws (extern "C"
//   entry, like the M <= 64 one), after r4d_gemm_trellis_nt_m256_check has already named the reason.
void r4d_gemm_trellis_nt_m256(int64_t a0, int64_t a1, int n_split, int64_t w, int64_t svh, int64_t c,
                              int64_t ws, int64_t tickets, int M, int K, int N, int KB, int SK, int NP,
                              int SKG, int U, float out_scale, int64_t stream, int skw);
// nullptr when (M, K, N, n_split, KB, SK, NP, SKG, U, skw) is a launch r4d_gemm_trellis_nt_m256 would
// accept, else a message (valid until this thread's next call) naming the first rule it breaks.
const char* r4d_gemm_trellis_nt_m256_check(int M, int K, int N, int n_split, int KB, int SK, int NP,
                                           int SKG, int U, int skw);
size_t r4d_gemm_trellis_nt_m256_ws_bytes(int M, int N, int SKG);   // SKG * M * N * 4

// ---- registry ------------------------------------------------------------------------------
// Every kernel in the library, with the constraints its name encodes spelled out. A caller that
// wants to know whether R4D covers a model can read this instead of hardcoding what it remembers.
//
// Each row carries the geometry twice: `shape` for a human, and `constraints` for a caller that
// wants to TEST a model against it. The second form is why the table exists -- without it every
// integration ends up restating the constraints in its own gate, and the copies drift.
//
// A constraint is one predicate on one named parameter. It is a NECESSARY condition, not a
// sufficient one: it settles whether a kernel exists for a geometry, never whether a particular
// call is safe. Strides, contiguity, alignment and buffer sizes are per-call and stay with the
// entry point, which still rejects anything it cannot run.
enum {
    R4D_C_EQ  = 0,  // param == ival
    R4D_C_LE  = 1,  // param <= ival
    R4D_C_GE  = 2,  // param >= ival
    R4D_C_DIV = 3,  // param % ival == 0
    R4D_C_IN  = 4,  // param is one of the space-separated words in sval
};
struct R4DConstraint {
    const char* key;   // parameter name, e.g. "head_dim"
    int         op;    // one of R4D_C_*
    long long   ival;  // numeric operand (EQ / LE / GE / DIV)
    const char* sval;  // string-set operand (IN)
};
struct R4DKernelInfo {
    const char* name;      // entry point, without the r4d_ prefix
    const char* family;    // attn | gdn | gemm | quant | dflash
    const char* op;        // the operation it implements -- the key callers select on, and the
                           // one thing several kernels may share (the GEMM families)
    const char* computes;  // what it computes, for a human
    const char* shape;     // the geometry it is compiled for, for a human
    const char* dtypes;    // operand dtypes
    const struct R4DConstraint* constraints;  // the same geometry, testable
    int n_constraints;
};
int r4d_kernel_count(void);
const struct R4DKernelInfo* r4d_kernel_at(int i);
}
