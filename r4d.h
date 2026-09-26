// R4D -- the RDNA4 (gfx1201) kernel library: attention, gated delta net, all-reduce and a skinny
// bf16 GEMM, compiled into one shared object. Plain C ABI: every entry point takes raw device
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
// all-reduce and GEMM throw std::runtime_error (they are only reachable from the pybind surface,
// which turns that into a Python exception at the call site).
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

// Library version. The module exposes it as r4d.__version__, and the git tag it was built from is
// expected to match -- which is what lets a consumer assert it linked the sources it pinned rather
// than whatever a stale clone happened to hold.
#define R4D_VERSION "0.5.0"

struct R4DArgs {
    const void*  q;             // (num_seqs*q_len, q_heads, head_dim)  bf16
    const void*  kv;            // (num_blocks, kv_heads, block_size, 2*head_dim)  fp8 e4m3 or bf16
                                //   strides below are in ELEMENTS of that dtype; K then V per slot
    const int*   block_table;   // (num_seqs, max_blocks)            int32
    const int*   seqused_k;     // (num_seqs,)                       int32
    void*        out;           // (total_q, q_heads, head_dim)      bf16
    const float* k_descale;     // (num_seqs, kv_heads)
    const float* v_descale;     // (num_seqs, kv_heads)
    const float* q_descale;     // unused: the query is bf16
    void*        scratch;       // split-KV partials (decode only), or null
    int num_seqs, q_len, q_heads, kv_heads, head_dim, block_size, max_blocks;
    int64_t kv_block_stride;       // elements between consecutive blocks
    int64_t kv_head_stride;        // elements between kv heads inside a block
    float scale;
    int  splits;                // decode only; 0 = let the split law choose
    int  max_ctx;               // host-visible context bound (seqused_k is device-side)
};

extern "C" {
// ---- attention: paged, causal, varlen ------------------------------------------------------
// Compiled for head_dim 256, 6 queries per KV head, paged block size 16, bf16 query. The prefill
// kernel tiles the query; the decode kernel splits the KV and takes at most 64 query rows
// (q_len * gqa), the band a speculative-decode verify step falls in.
// Return 0 on success, negative on a shape this instantiation does not serve.
int  r4d_attn_prefill_h256_gqa6_fp8kv (const R4DArgs* a, hipStream_t stream);
int  r4d_attn_prefill_h256_gqa6_bf16kv(const R4DArgs* a, hipStream_t stream);
int  r4d_attn_decode_h256_gqa6_fp8kv  (const R4DArgs* a, hipStream_t stream);
int  r4d_attn_decode_h256_gqa6_bf16kv (const R4DArgs* a, hipStream_t stream);
// Bytes of split-KV partial buffer one decode launch of this shape needs. Independent of the cache
// dtype: the partials are f16 either way.
int64_t r4d_attn_decode_h256_gqa6_scratch_bytes(const R4DArgs* a);
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

// ---- all-reduce: one-shot, push, 2 ranks over P2P ------------------------------------------
// One-shot means each rank pushes its whole input into the peer's IPC scratch and then reduces
// locally -- no ring, no two-shot reduce-scatter, so the scratch is sized by the full message.
// Exactly 2 ranks: the handshake is a single peer flag, not a tree.
enum { R4D_AR_HANDLE_BYTES = 64 };
// IPC scratch helpers. Startup-only, and not kernels: alloc returns the device pointer and writes
// the IPC handle (up to HANDLE_BYTES) into out_handle.
int64_t r4d_ar_ipc_alloc(int64_t size, int finegrained, char* out_handle, int* out_len);
int64_t r4d_ar_ipc_open(const char* handle, int len);
void r4d_ar_ipc_free(int64_t p);
void r4d_ar_ipc_memzero(int64_t p, int64_t size);
void r4d_ar_ipc_enable_peer(int64_t peer);
// Exact sum, fp32 accumulate, bf16 / fp16 / fp32 payload (dtype 0 / 1 / 2).
void r4d_ar_oneshot_2rank_exact(int64_t peer_scratch, int64_t my_scratch, int64_t peer_flags, int64_t my_flags,
                                int64_t seq_ctrs, int64_t slot_stride16, int64_t inp, int64_t out, int64_t n_elem,
                                int64_t dtype, int64_t stream, int64_t nblocks, int64_t nthreads, int64_t drain,
                                int64_t acq);
// Same topology, but the wire payload is Walsh-Hadamard rotated and quantised to 6 bits per element
// over groups of 64 (plus a bf16 scale per group). Lossy, and bf16 / fp16 payload only. Takes this
// rank's own packed copy (loc_pack) as well, so the reduce folds exactly the bytes it sent.
void r4d_ar_oneshot_2rank_wht6(int64_t peer_scratch, int64_t my_scratch, int64_t peer_flags, int64_t my_flags,
                               int64_t seq_ctrs, int64_t loc_pack, int64_t slot_stride_bytes,
                               int64_t scale_off_bytes, int64_t inp, int64_t out, int64_t n_elem, int64_t dtype,
                               int64_t stream, int64_t nblocks, int64_t nthreads, int64_t drain, int64_t acq);
int  r4d_ar_max_blocks(void);                                   // the two 2-rank kernels
void r4d_ar_wht6_dims(int* group, int* bits, int* chunk_elems); // rotated-6-bit payload only

// ---- all-reduce: one-shot, push, 4 or 8 ranks over P2P -------------------------------------
// The same algorithm at width (r4d_ar_oneshot_Nrank_exact.hip). The peer arguments are ascending
// global rank with the current rank removed. Scratch is 2*world size slots of slot_stride16 16B 
// words parity x source rank, this rank's own slot unused); flags are max_blocks*ws words indexed 
// [block][source rank]; seq_ctrs is one word per block.
void r4d_ar_oneshot_4rank_exact(int64_t peer_scratch0, int64_t peer_scratch1, int64_t peer_scratch2,
                                int64_t peer_flags0, int64_t peer_flags1, int64_t peer_flags2,
                                int64_t my_scratch, int64_t my_flags, int64_t seq_ctrs, int64_t slot_stride16,
                                int64_t inp, int64_t out, int64_t n_elem, int64_t dtype, int64_t rank,
                                int64_t stream, int64_t nblocks, int64_t nthreads, int64_t drain, int64_t acq,
                                int64_t pub);
void r4d_ar_oneshot_8rank_exact(int64_t peer_scratch0, int64_t peer_scratch1, int64_t peer_scratch2,
                                int64_t peer_scratch3, int64_t peer_scratch4, int64_t peer_scratch5,
                                int64_t peer_scratch6, int64_t peer_flags0, int64_t peer_flags1,
                                int64_t peer_flags2, int64_t peer_flags3, int64_t peer_flags4,
                                int64_t peer_flags5, int64_t peer_flags6, int64_t my_scratch,
                                int64_t my_flags, int64_t seq_ctrs, int64_t slot_stride16, int64_t inp,
                                int64_t out, int64_t n_elem, int64_t dtype, int64_t rank, int64_t stream,
                                int64_t nblocks, int64_t nthreads, int64_t drain, int64_t acq, int64_t pub);
void r4d_ar_wide_dims(int* max_blocks, int* nb_design, int* max_peers);

// ---- all-reduce: two-shot, 4 or 8 ranks over P2P -------------------------------------------
// Reduce-scatter + all-gather on the same buffer trio and peer-argument order as the wide
// one-shot (r4d_ar_twoshot_Nrank_exact.hip): shard s, a contiguous 1/ws of the message, is
// reduced by rank s (fp32, ascending rank order), then gathered. Sends 2*(ws-1)/ws*N bytes
// per rank in two hops. Covers the sizes where the one-shot's (ws-1)*N send is too heavy.
// Regions A
// and B at 0 and shard16, so the slot stride must hold 2*shard16. Above this bound the caller
// falls back:
enum { R4D_AR_TWOSHOT_WIDE_MAX_ELEMS = 20971520 };  // a 4096-token hidden-5120 prefill chunk
void r4d_ar_twoshot_4rank_exact(int64_t peer_scratch0, int64_t peer_scratch1, int64_t peer_scratch2,
                                int64_t peer_flags0, int64_t peer_flags1, int64_t peer_flags2,
                                int64_t my_scratch, int64_t my_flags, int64_t seq_ctrs,
                                int64_t slot_stride16, int64_t inp, int64_t out, int64_t n_elem, int64_t dtype,
                                int64_t rank, int64_t stream, int64_t nblocks, int64_t nthreads, int64_t drain,
                                int64_t acq, int64_t pub);
void r4d_ar_twoshot_8rank_exact(int64_t peer_scratch0, int64_t peer_scratch1, int64_t peer_scratch2,
                                int64_t peer_scratch3, int64_t peer_scratch4, int64_t peer_scratch5,
                                int64_t peer_scratch6, int64_t peer_flags0, int64_t peer_flags1,
                                int64_t peer_flags2, int64_t peer_flags3, int64_t peer_flags4,
                                int64_t peer_flags5, int64_t peer_flags6, int64_t my_scratch,
                                int64_t my_flags, int64_t seq_ctrs, int64_t slot_stride16, int64_t inp,
                                int64_t out, int64_t n_elem, int64_t dtype, int64_t rank, int64_t stream,
                                int64_t nblocks, int64_t nthreads, int64_t drain, int64_t acq, int64_t pub);
// The tiered int8 wire on both of the two-shot's hops (r4d_ar_twoshot_Nrank_ti8.hip; codec
// in r4d_ar_ti8.h), 4 ranks only. The wire: int8 groups of 32 with an fp16 scale and a
// per-group record (T1 plain, T2 top-1-exact, T3 wht32) 38 bytes/group, deterministic
// tiering, bit-identical across ranks. Same peer-argument order and flag layout as the exact
// pair; slot stride in bytes (the wire is not 16B-aligned), holding two wire images; the 
// encoder writes its own slot too (no loc_pack). numel must tile 32-elem groups over the ws
// shards (numel % 128).
void r4d_ar_twoshot_4rank_ti8(int64_t peer_scratch0, int64_t peer_scratch1, int64_t peer_scratch2,
                              int64_t peer_flags0, int64_t peer_flags1, int64_t peer_flags2,
                              int64_t my_scratch, int64_t my_flags, int64_t seq_ctrs,
                              int64_t slot_stride_bytes, int64_t inp, int64_t out, int64_t n_elem,
                              int64_t dtype, int64_t rank, int64_t stream, int64_t nblocks,
                              int64_t nthreads, int64_t drain, int64_t acq, int64_t pub);
void r4d_ar_ti8_dims(int* group, int* group_bytes);  // tiered-int8 wire only

// ---- GEMM ----------------------------------------------------------------------------------
// C[M,N] = A[M,K] @ W[N,K]^T, bf16 throughout: a torch Linear with the weight stored (N,K), which
// is the "nt" in the name. Specialised for skinny M -- M <= 16, the band a small projection such
// as an MoE router gate produces -- where the weight read dominates and rocBLAS does poorly.
// WV columns per block x SK k-splits, reduced in LDS.
void r4d_gemm_bf16_nt_m16(int64_t a, int64_t w, int64_t c, int M, int K, int N, int WV, int SK,
                          int64_t stream);
int  r4d_gemm_bf16_nt_m16_max_m(void);

// Skinny bf16 GEMM for M up to 64: the same C[M,N] = A[M,K] @ W[N,K]^T, computed with
// 16x16x16 WMMA so a 16-wide step of K costs ceil(M/16) activation fragments and one weight
// fragment, instead of one weight load and M activation loads.
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
// The same kernel at a group chosen per call. has_group() is 1 for 32, 64 and 128, 0 otherwise --
// a build default outside those three is served by r4d_gemm_w4a16_nt_m64 only. The packed Wq is
// the same bytes at every group; Wsz has K/group dwords per row. K must be divisible by SK * max(group, 64) -- a split has to start on a 64-K packed
// block -- and an uninstantiated group throws, like any other rejected shape.
int  r4d_gemm_w4a16_nt_m64_has_group(int group);
void r4d_gemm_w4a16_nt_m64_g(int group, int64_t a, int64_t wq, int64_t wsz, int64_t c, int M, int K,
                             int N, int WV, int SK, int MB, int NPW, int NT, int64_t stream);

// Skinny GEMM with a TRELLIS-coded weight: EXL3 / QTIP tiles of 16 k x 16 n, each a tail-biting
// ring of 256*KB bits whose 16-bit states hash to f16 values (the "mul1" codebook), f16 A, 16x16x16
// WMMA, one kernel for M = 1..64. `w` is the tiles' uint32 ring words in the PAIR GRID -- word w of
// tile (tn, tk) at (((tn >> 1) * (K/16) + tk) * 2 + (tn & 1)) * 8KB + w -- N*K*KB/8 bytes, 16-byte
// aligned. Q[K,N] below is the decoded weight in its regularized domain; the linear it belongs to is
// W^T = diag(suh) H Q H diag(svh) (128-point Hadamards), whose input side is applied to A before
// this call. Two A parts: output columns >= n_split read a1 (a fused gate/up pair with different
// input transforms), all others a0; a1 = 0 means a0. Tuning: WV x SK waves per block (column blocks
// x in-block K splits), NP tile pairs per wave (block width Wc = 32 WV NP, one of 32/64/128/256),
// SKG blocks splitting K across the grid, MT row tiles per block, U k-tiles per step, NT for
// non-temporal weight loads. Legal only when K and N are multiples of 128, N and n_split of Wc,
// (K/16) of SK*SKG*U, SK*Wc*32 <= 64 KiB, WV*SK*32 <= 1024, and (NP, U, MT) is instantiated
// (NP, U in {1,2,4}, NP*U <= 8; MT up to 4 at NP 1 and at (NP, U) = (2, 1), 3 at (2, 2) and
// (2, 4), 2 at (4, 1), 1 at (4, 2) -- what fits 190 VGPRs); anything else throws. A 128-column
// group whose sums come from
// more than one block (SKG > 1 or Wc < 128) is finished by the last block to arrive, through `ws`
// (fp32, ws_bytes(M, N, SKG)) and `tickets` (u32 [N/128], zero before the first call; every
// complete launch leaves them zero again), and needs every row tile in one block (ceil(M/16) <= MT).
// A row's result depends on the tuning's summation order (SK, SKG, Wc), never on M or on the other
// rows, so a caller that gives every M <= 16 the same tuning gets verify rows equal to decode rows.
//
// _raw is the test and diagnostic entry: fp32 C[M,N] = A @ Q with no output transform, so one-hot
// A rows return rows of Q bit for bit. `clk` (0 = off): 4 uint64 that block (0,0,0)'s thread 0
// fills with clock64() / wall_clock64() at entry and at the end of its own work, for the shader
// clock during a real launch.
void r4d_gemm_trellis_nt_m64_raw(int64_t a0, int64_t a1, int n_split, int64_t w, int64_t c,
                                 int64_t ws, int64_t tickets, int M, int K, int N, int KB, int WV,
                                 int SK, int MT, int NP, int SKG, int U, int NT, int64_t clk,
                                 int64_t stream);
int    r4d_gemm_trellis_nt_m64_has_rate(int KB);   // 1 for an instantiated KB (4), else 0
int    r4d_gemm_trellis_nt_m64_max_m(void);         // 64
size_t r4d_gemm_trellis_nt_m64_ws_bytes(int M, int N, int SKG);
// The whole decoded Q[K,N] as f16 bits (row-major, K rows), from the same pair-grid words and the
// same device decode as the GEMM, bit-exact against the format's reference decode. KB 4 or 5; K a
// multiple of 16, N of 32.
void   r4d_trellis_reconstruct_f16(int64_t w, int64_t q, int K, int N, int KB, int64_t stream);

// 4-bit weight, 8-bit activation. Signed 4-bit codes, per-row activation scale, int8 WMMA.
void r4d_gemm_w4a8_nt_m64(int64_t a, int64_t ascale, int64_t wq, int64_t ws, int64_t c, int M, int K, int N,
                          int WV, int SK, int MB, int NPW, int NT, int64_t stream);
int  r4d_gemm_w4a8_nt_m64_max_m(void);
int  r4d_gemm_w4a8_nt_m64_group(void);
int  r4d_gemm_w4a8_nt_m64_aperm(void);

// OCP-MXFP4 weight, fp8 activation. e2m1 elements with one E8M0 exponent per 32 K, folded
// against a per-row reference exponent so the inner loop has no rescale; fp8 WMMA.
void r4d_gemm_mxfp4a8_nt_m64(int64_t a, int64_t ascale, int64_t wq, int64_t ws, int64_t wref, int64_t c,
                             int M, int K, int N, int WV, int SK, int MB, int NPW, int64_t stream);
int  r4d_gemm_mxfp4a8_nt_m64_max_m(void);
int  r4d_gemm_mxfp4a8_nt_m64_group(void);

// Per-row symmetric int8 quantisation of a bf16 activation, in the A-fragment byte order.
void r4d_quant_act_i8(int64_t a, int64_t q, int64_t s, int M, int K, int64_t stream);

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
    const char* family;    // attn | gdn | ar | gemm
    const char* op;        // the operation it implements -- the key callers select on, and the
                           // one thing several kernels may share (fp8 and bf16 KV, exact and
                           // lossy all-reduce)
    const char* computes;  // what it computes, for a human
    const char* shape;     // the geometry it is compiled for, for a human
    const char* dtypes;    // operand dtypes
    const struct R4DConstraint* constraints;  // the same geometry, testable
    int n_constraints;
};
int r4d_kernel_count(void);
const struct R4DKernelInfo* r4d_kernel_at(int i);
}
