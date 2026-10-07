// r4dx::core::r4d -- thin typed C++ wrappers over the r4d.h C ABI (third_party/libr4d/r4d.h,
// windows-llp64 @ 7675605). Every wrapper here:
//   - takes typed pointers / r4dx::core::TensorView-shaped arguments instead of raw int64_t,
//   - documents strides in ELEMENTS and dtypes in the argument comment, mirroring r4d.h itself,
//   - converts a negative r4d return code (attn/gdn family) into an r4dx::core::R4dError naming
//     the kernel, via R4DX_R4D_CHECK,
//   - takes an r4dx::core::Stream (or a raw hipStream_t) rather than an opaque int64_t/void*.
// r4d.h's `int64_t` device-pointer convention is preserved at the call boundary
// (reinterpret_cast<int64_t>(ptr)) because that is the actual r4d C ABI; only r4dx-side code
// gets the typed wrapper.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <vector>

#include "r4d.h"
#include "r4dx/core/error.hpp"

namespace r4dx::core::r4d {

// ---- geometry ---------------------------------------------------------------------------
struct AttnDims {
  int head_dim, gqa, block_size, max_decode_rows;
};
inline AttnDims GetAttnDims() {
  AttnDims d{};
  r4d_attn_dims(&d.head_dim, &d.gqa, &d.block_size, &d.max_decode_rows);
  return d;
}

struct GdnDims {
  int head_k, head_v, chunk;
};
inline GdnDims GetGdnDims() {
  GdnDims d{};
  r4d_gdn_dims(&d.head_k, &d.head_v, &d.chunk);
  return d;
}

struct AttnVitDims {
  int head_dim, rows_large, rows_small, split;
};
inline AttnVitDims GetAttnVitDims() {
  AttnVitDims d{};
  r4d_attn_vit_dims(&d.head_dim, &d.rows_large, &d.rows_small, &d.split);
  return d;
}

// ---- attention: paged, causal, varlen --------------------------------------------------
// Mirrors R4DArgs (r4d.h:42-59) field for field; construct with designated-initializer style
// (C++20 not required -- plain aggregate init works) and pass to one of the call wrappers below.
//
// Args::k_descale / Args::v_descale shape: r4d_attn_decode_h256_gqa6.hip indexes these as
// `seq * kv_heads + kvh`, i.e. a [num_seqs, kv_heads] table -- NOT the [kv_heads] single-row
// table r4dx_kv_write_paged_fp8_hnd takes (kernels.h). A caller with num_seqs > 1 must broadcast
// the per-layer [kv_heads] descale row num_seqs times before building Args; passing a [kv_heads]
// buffer as-is (as tests/kernels/test_attn_decode.cpp's num_seqs==1 case does, where the two
// shapes coincide) reads out-of-bounds/garbage descales for sequences 1..num_seqs-1.
using Args = R4DArgs;

inline void AttnPrefillFp8Kv(const Args& a, hipStream_t stream) {
  R4DX_R4D_CHECK("attn_prefill_h256_gqa6_fp8kv", r4d_attn_prefill_h256_gqa6_fp8kv(&a, stream));
}
// Split-KV prefill (prefill M1): Args::splits segments of the KV range, fp32 partials in
// Args::scratch (AttnPrefillSplitKvScratchBytes), fixed-order merge. splits <= 1 is exactly
// AttnPrefill*Kv above (same kernel, same bits, no scratch).
inline void AttnPrefillSplitKvFp8Kv(const Args& a, hipStream_t stream) {
  R4DX_R4D_CHECK("attn_prefill_splitkv_h256_gqa6_fp8kv",
                 r4d_attn_prefill_splitkv_h256_gqa6_fp8kv(&a, stream));
}
inline int64_t AttnPrefillSplitKvScratchBytes(const Args& a) {
  return r4d_attn_prefill_splitkv_h256_gqa6_scratch_bytes(&a);
}
// Exact-wide prefill (prefill M1 lossless mode): AttnPrefillFp8Kv's bits over more workgroups.
// Args::splits 0 = the default geometry (see r4d.h).
inline void AttnPrefillExactFp8Kv(const Args& a, hipStream_t stream) {
  R4DX_R4D_CHECK("attn_prefill_exact_h256_gqa6_fp8kv",
                 r4d_attn_prefill_exact_h256_gqa6_fp8kv(&a, stream));
}
inline void AttnDecodeFp8Kv(const Args& a, hipStream_t stream) {
  R4DX_R4D_CHECK("attn_decode_h256_gqa6_fp8kv", r4d_attn_decode_h256_gqa6_fp8kv(&a, stream));
}
// Bytes of split-KV decode scratch this shape needs; allocate and set Args::scratch before
// calling AttnDecode*.
inline int64_t AttnDecodeScratchBytes(const Args& a) {
  return r4d_attn_decode_h256_gqa6_scratch_bytes(&a);
}

// ---- attention: sliding window (Gemma 4's sliding layers; r4d_attn_paged_h256_gqa2.hip) ------
// ArgsW = Args + the window and the prefill-only klimit_ext (r4d.h's R4DArgsW). head_dim 256, 2 q per
// kv head, block 16, fp8 KV; a sliding KV ring is a block table whose entries repeat, and its cache must
// be zero-initialised (r4d.h). Decode needs ArgsW::scratch (AttnDecodeWindowScratchBytes); a negative
// return throws, naming the kernel.
using ArgsW = R4DArgsW;

inline void AttnPrefillWindowFp8Kv(const ArgsW& a, hipStream_t stream) {
  R4DX_R4D_CHECK("attn_prefill_h256_gqa2_fp8kv", r4d_attn_prefill_h256_gqa2_fp8kv(&a, stream));
}
inline void AttnDecodeWindowFp8Kv(const ArgsW& a, hipStream_t stream) {
  R4DX_R4D_CHECK("attn_decode_h256_gqa2_fp8kv", r4d_attn_decode_h256_gqa2_fp8kv(&a, stream));
}
inline int64_t AttnDecodeWindowScratchBytes(const ArgsW& a) {
  return r4d_attn_decode_h256_gqa2_scratch_bytes(&a);
}

// Every attention geometry the build serves (r4d_attn_dims reports the gqa6 one only), for validating
// a model's layer types against all of them at load.
inline std::vector<R4DAttnGeom> GetAttnGeoms() {
  std::vector<R4DAttnGeom> g;
  for (int i = 0; i < r4d_attn_geom_count(); ++i) g.push_back(*r4d_attn_geom_at(i));
  return g;
}

// ---- attention: vision encoder ----------------------------------------------------------
// q,k,v,o: [total_tokens, heads, head_dim] bf16, contiguous. cu_seqlens: device int32[num_seqs+1].
inline void AttnVitBf16(const void* q, const void* k, const void* v, void* o,
                         const void* cu_seqlens, int num_seqs, int max_seqlen, int heads,
                         int head_dim, float scale, hipStream_t stream) {
  R4DX_R4D_CHECK("attn_vit_h72_bf16",
                 r4d_attn_vit_h72_bf16(q, k, v, o, cu_seqlens, num_seqs, max_seqlen, heads,
                                       head_dim, scale, stream));
}

// ---- gated delta net ----------------------------------------------------------------------
// q,k [T,Hg,K] bf16; v,o [T,H,V] bf16; A [T,H,bt] bf16; g,beta [T,H] fp32; h0,ht [N,H,V,K] fp32;
// cu [N+1] int32. K=128, V=128, bt=64 (r4d_gdn_dims()); a mismatch throws.
inline void GdnChunkScan(const void* q, const void* k, const void* v, const void* A,
                          const void* g, const void* beta, const void* h0, void* o, void* ht,
                          const void* cu, int N, int H, int Hg, int K, int V, int bt, float scale,
                          hipStream_t stream) {
  R4DX_R4D_CHECK("gdn_chunk_scan_k128_v128_c64_bf16",
                 r4d_gdn_chunk_scan_k128_v128_c64_bf16(q, k, v, A, g, beta, h0, o, ht, cu, N, H,
                                                        Hg, K, V, bt, scale, stream));
}

// k [T,Hg,K] bf16; beta,g [T,H] fp32 (g already chunk-local-cumsum'd); A [T,H,64] bf16 out;
// cu [N+1] int32.
inline void GdnKktSolve(const void* k, const void* beta, const void* g, void* A, const void* cu,
                         int N, int T, int H, int Hg, int K, int bt, hipStream_t stream) {
  R4DX_R4D_CHECK("gdn_kkt_solve_k128_c64_bf16",
                 r4d_gdn_kkt_solve_k128_c64_bf16(k, beta, g, A, cu, N, T, H, Hg, K, bt, stream));
}

inline void GdnGatedRmsNorm(const void* x, const void* z, const void* w, void* o, int64_t rows,
                             int64_t xrow, int64_t zrow, int64_t orow, int width, float eps,
                             int act, hipStream_t stream) {
  R4DX_R4D_CHECK("gdn_gated_rmsnorm_h128_bf16",
                 r4d_gdn_gated_rmsnorm_h128_bf16(x, z, w, o, rows, xrow, zrow, orow, width, eps,
                                                  act, stream));
}

inline void GdnRecurrentUpdate(const void* q, const void* k, const void* v, const void* a,
                                const void* b, int64_t ab_stride, int ab_is_bf16,
                                const void* A_log, const void* dt_bias, void* state,
                                int64_t state_slot_stride, int64_t state_head_stride, void* o,
                                const void* cu, const void* ssm_state_indices,
                                int64_t indices_stride, const void* num_accepted,
                                const void* z_gate, const void* norm_weight, float norm_eps,
                                int norm_act, int N, int H, int Hg, int K, int V, float scale,
                                float softplus_thr, hipStream_t stream) {
  R4DX_R4D_CHECK(
      "gdn_recurrent_update_k128_v128_bf16_fp32state",
      r4d_gdn_recurrent_update_k128_v128_bf16_fp32state(
          q, k, v, a, b, ab_stride, ab_is_bf16, A_log, dt_bias, state, state_slot_stride,
          state_head_stride, o, cu, ssm_state_indices, indices_stride, num_accepted, z_gate,
          norm_weight, norm_eps, norm_act, N, H, Hg, K, V, scale, softplus_thr, stream));
}

inline void GdnConvPrep(const void* x, int64_t xpitch, const void* wgt, const void* bias,
                         void* cstate, int64_t cs_seq, int64_t cs_dim, int64_t cs_tok,
                         const void* cache_idx, int64_t ci_stride, const void* has_init,
                         const void* a, const void* b, int64_t ab_stride, int ab_is_bf16,
                         const void* A_log, const void* dt_bias, void* q, void* k, void* v,
                         void* g, void* beta, const void* cu, int N, int T, int H, int Hg, int K,
                         int V, int width, float softplus_thr, hipStream_t stream) {
  R4DX_R4D_CHECK("gdn_conv_prep_w4_h128_bf16",
                 r4d_gdn_conv_prep_w4_h128_bf16(x, xpitch, wgt, bias, cstate, cs_seq, cs_dim,
                                                 cs_tok, cache_idx, ci_stride, has_init, a, b,
                                                 ab_stride, ab_is_bf16, A_log, dt_bias, q, k, v, g,
                                                 beta, cu, N, T, H, Hg, K, V, width, softplus_thr,
                                                 stream));
}

// GdnConvPrep's outputs bit for bit, on a grid that fills the device (r4d.h's
// r4d_gdn_conv_prep2_w4_h128_bf16). Same arguments.
inline void GdnConvPrep2(const void* x, int64_t xpitch, const void* wgt, const void* bias,
                          void* cstate, int64_t cs_seq, int64_t cs_dim, int64_t cs_tok,
                          const void* cache_idx, int64_t ci_stride, const void* has_init,
                          const void* a, const void* b, int64_t ab_stride, int ab_is_bf16,
                          const void* A_log, const void* dt_bias, void* q, void* k, void* v,
                          void* g, void* beta, const void* cu, int N, int T, int H, int Hg, int K,
                          int V, int width, float softplus_thr, hipStream_t stream) {
  R4DX_R4D_CHECK("gdn_conv_prep2_w4_h128_bf16",
                 r4d_gdn_conv_prep2_w4_h128_bf16(x, xpitch, wgt, bias, cstate, cs_seq, cs_dim,
                                                  cs_tok, cache_idx, ci_stride, has_init, a, b,
                                                  ab_stride, ab_is_bf16, A_log, dt_bias, q, k, v, g,
                                                  beta, cu, N, T, H, Hg, K, V, width, softplus_thr,
                                                  stream));
}

inline void GdnConvUpdate(const void* x, int64_t xpitch, const void* wgt, const void* bias,
                           void* cstate, int64_t cs_seq, int64_t cs_dim, int64_t cs_tok,
                           int state_len_max, const void* cache_idx, int64_t ci_stride,
                           const void* num_accepted, void* q, void* k, void* v, const void* cu,
                           int N, int H, int Hg, int K, int V, int width, int max_query_len,
                           hipStream_t stream) {
  R4DX_R4D_CHECK("gdn_conv_update_w4_h128_bf16",
                 r4d_gdn_conv_update_w4_h128_bf16(x, xpitch, wgt, bias, cstate, cs_seq, cs_dim,
                                                   cs_tok, state_len_max, cache_idx, ci_stride,
                                                   num_accepted, q, k, v, cu, N, H, Hg, K, V,
                                                   width, max_query_len, stream));
}

// GDN state sizes (elements), for src/core allocators. Recurrent state is [N,H,V,K] fp32
// (docs/architecture.md "GDN state"); conv state is a rolling buffer of (kernel-1) + spec-window
// entries per (sequence, channel).
inline int64_t GdnRecurrentStateElems(int64_t num_seqs, int64_t H, int64_t V, int64_t K) {
  return num_seqs * H * V * K;
}
inline int64_t GdnConvStateElems(int64_t num_seqs, int64_t conv_dim, int64_t width,
                                  int64_t spec_window = 0) {
  return num_seqs * conv_dim * (width - 1 + spec_window);
}

// ---- GEMM -----------------------------------------------------------------------------------
inline void GemmBf16NtM64(const void* a, const void* w, void* c, int M, int K, int N, int WV,
                           int SK, int MB, hipStream_t stream) {
  r4d_gemm_bf16_nt_m64(reinterpret_cast<int64_t>(a), reinterpret_cast<int64_t>(w),
                        reinterpret_cast<int64_t>(c), M, K, N, WV, SK, MB,
                        reinterpret_cast<int64_t>(stream));
}
inline void GemmW4a16NtM64(const void* a, const void* wq, const void* wsz, void* c, int M, int K,
                            int N, int WV, int SK, int MB, int NPW, int NT, hipStream_t stream) {
  r4d_gemm_w4a16_nt_m64(reinterpret_cast<int64_t>(a), reinterpret_cast<int64_t>(wq),
                         reinterpret_cast<int64_t>(wsz), reinterpret_cast<int64_t>(c), M, K, N,
                         WV, SK, MB, NPW, NT, reinterpret_cast<int64_t>(stream));
}
// The same GEMM at a per-tensor w4a16 group (docs/quant2.md section 5.1): 32 or 64. K must be
// divisible by SK * max(group, 64); the kernel throws otherwise.
inline void GemmW4a16NtM64G(int group, const void* a, const void* wq, const void* wsz, void* c,
                             int M, int K, int N, int WV, int SK, int MB, int NPW, int NT,
                             hipStream_t stream) {
  r4d_gemm_w4a16_nt_m64_g(group, reinterpret_cast<int64_t>(a), reinterpret_cast<int64_t>(wq),
                           reinterpret_cast<int64_t>(wsz), reinterpret_cast<int64_t>(c), M, K, N,
                           WV, SK, MB, NPW, NT, reinterpret_cast<int64_t>(stream));
}
// Trellis-coded weight (docs/trellis-kernel.md 4.3-4.5; r4d.h has the whole contract): C[M][N]
// bf16 = bf16((FWHT128(A_p @ Q)[m][n] * svh[n]) * out_scale). a0 / a1: f16 [M][K], row stride K,
// ALREADY input-transformed (r4dx_trellis_input_bf16); output columns >= n_split read a1 (a1
// nullptr: a0 everywhere, n_split N for a one-part linear). w: pair-grid ring words; svh: fp32
// [N]; out_scale = 2^-s / sqrt(128). ws (fp32, GemmTrellisWsBytes) and tickets (u32 [N/128],
// GemmTrellisTicketsBytes) serve a 128-column group split across blocks (SKG > 1 or a block
// narrower than 128 columns) and may be nullptr otherwise. MT is LinearTuning::MB and NP
// LinearTuning::NPW. Throws on an illegal shape or tuning.
inline void GemmTrellisNtM64(const void* a0, const void* a1, int n_split, const void* w,
                              const void* svh, void* c, void* ws, void* tickets, int M, int K,
                              int N, int KB, int WV, int SK, int MT, int NP, int SKG, int U, int NT,
                              float out_scale, hipStream_t stream) {
  r4d_gemm_trellis_nt_m64(reinterpret_cast<int64_t>(a0), reinterpret_cast<int64_t>(a1), n_split,
                          reinterpret_cast<int64_t>(w), reinterpret_cast<int64_t>(svh),
                          reinterpret_cast<int64_t>(c), reinterpret_cast<int64_t>(ws),
                          reinterpret_cast<int64_t>(tickets), M, K, N, KB, WV, SK, MT, NP, SKG, U,
                          NT, out_scale, reinterpret_cast<int64_t>(stream));
}
// SKG * M * N fp32: the split-group partials of one M-row call.
inline size_t GemmTrellisWsBytes(int M, int N, int SKG) {
  return r4d_gemm_trellis_nt_m64_ws_bytes(M, N, SKG);
}
// (N / 128) u32: one linear's tickets.
inline size_t GemmTrellisTicketsBytes(int N) { return r4d_gemm_trellis_nt_m64_tickets_bytes(N); }
// hipMemsetAsync(tickets, 0, bytes) on `stream`: the reset after a launch that did not complete.
inline void GemmTrellisZeroTickets(void* tickets, size_t bytes, hipStream_t stream) {
  r4d_gemm_trellis_nt_m64_zero_tickets(reinterpret_cast<int64_t>(tickets), bytes,
                                       reinterpret_cast<int64_t>(stream));
}
// M = 256 (or 128) trellis GEMM (r4d.h: r4d_gemm_trellis_nt_m256; docs/trellis-m256.md): the same
// linear as GemmTrellisNtM64 for M rows in one launch, bit-identical to what the M <= 64 kernel gives
// each 64-row slice at the same (SK, SKG). `ws` (GemmTrellisM256WsBytes) is always required. SK, SKG
// are the M = 64 tuning row's, NP / U the kernel's own ((1, 4) for M = 256), skw the K slices
// resident per workgroup (0 = min(SK, 4)). Throws on an illegal or uninstantiated combination.
inline void GemmTrellisNtM256(const void* a0, const void* a1, int n_split, const void* w,
                               const void* svh, void* c, void* ws, void* tickets, int M, int K, int N,
                               int KB, int SK, int NP, int SKG, int U, float out_scale, int skw,
                               hipStream_t stream) {
  r4d_gemm_trellis_nt_m256(reinterpret_cast<int64_t>(a0), reinterpret_cast<int64_t>(a1), n_split,
                           reinterpret_cast<int64_t>(w), reinterpret_cast<int64_t>(svh),
                           reinterpret_cast<int64_t>(c), reinterpret_cast<int64_t>(ws),
                           reinterpret_cast<int64_t>(tickets), M, K, N, KB, SK, NP, SKG, U, out_scale,
                           reinterpret_cast<int64_t>(stream), skw);
}
// nullptr when the launch is legal, else the message naming the first rule it breaks (valid until
// this thread's next call).
inline const char* GemmTrellisM256Check(int M, int K, int N, int n_split, int KB, int SK, int NP,
                                         int SKG, int U, int skw) {
  return r4d_gemm_trellis_nt_m256_check(M, K, N, n_split, KB, SK, NP, SKG, U, skw);
}
// SKG * M * N fp32.
inline size_t GemmTrellisM256WsBytes(int M, int N, int SKG) {
  return r4d_gemm_trellis_nt_m256_ws_bytes(M, N, SKG);
}

// The int8 x int8 trellis prefill GEMM (r4d.h "int8 x int8 trellis prefill GEMM", R4DX_PREFILL_INT8,
// docs/int8-prefill.md "Production path"). TrellisI8WscaleCount floats are the weight-scale table of one trellis
// linear ([K / 128][N] fp32), built once from its words; TrellisI8QuantAct quantizes a transformed activation
// ([256][K] f16 per part) to A8 (256 K bytes per part) + SA ((K / 128) * 256 floats per part), the parts laid
// out one after the other; GemmTrellisNtI8 is the linear for exactly 256 rows (a1 / sa1 null: one part, then
// n_split = N). Throws on an illegal combination, GemmTrellisNtI8Check names the first rule it breaks (nullptr
// when legal).
inline size_t TrellisI8WscaleCount(int K, int N) { return r4d_trellis_i8_wscale_count(K, N); }
inline void TrellisI8WscaleBuild(const void* w, void* sw, int K, int N, int KB, hipStream_t stream) {
  r4d_trellis_i8_wscale(reinterpret_cast<int64_t>(w), reinterpret_cast<int64_t>(sw), K, N, KB,
                        reinterpret_cast<int64_t>(stream));
}
// Diagnostic (tests): the int8 matrix the GEMM quantizes on the fly -- block layout in w8 (N K bytes) and, if
// wp is not null, plain [N][K].
inline void TrellisI8DumpW(const void* w, const void* sw, void* w8, void* wp, int K, int N, int KB,
                           hipStream_t stream) {
  r4d_trellis_i8_dump_w(reinterpret_cast<int64_t>(w), reinterpret_cast<int64_t>(sw), reinterpret_cast<int64_t>(w8),
                        reinterpret_cast<int64_t>(wp), K, N, KB, reinterpret_cast<int64_t>(stream));
}
inline void TrellisI8QuantAct(const void* x, void* a8, void* sa, int parts, int64_t part_stride, int K,
                              hipStream_t stream) {
  r4d_trellis_i8_quant_act(reinterpret_cast<int64_t>(x), reinterpret_cast<int64_t>(a8),
                           reinterpret_cast<int64_t>(sa), parts, part_stride, K, reinterpret_cast<int64_t>(stream));
}
inline void GemmTrellisNtI8(const void* a8_0, const void* sa_0, const void* a8_1, const void* sa_1, int n_split,
                            const void* w, const void* sw, const void* svh, void* c, void* ws, void* tickets,
                            int M, int K, int N, int KB, int skw, int skg, float out_scale, hipStream_t stream) {
  r4d_gemm_trellis_nt_i8(reinterpret_cast<int64_t>(a8_0), reinterpret_cast<int64_t>(sa_0),
                         reinterpret_cast<int64_t>(a8_1), reinterpret_cast<int64_t>(sa_1), n_split,
                         reinterpret_cast<int64_t>(w), reinterpret_cast<int64_t>(sw), reinterpret_cast<int64_t>(svh),
                         reinterpret_cast<int64_t>(c), reinterpret_cast<int64_t>(ws),
                         reinterpret_cast<int64_t>(tickets), M, K, N, KB, skw, skg, out_scale,
                         reinterpret_cast<int64_t>(stream));
}
inline const char* GemmTrellisNtI8Check(int M, int K, int N, int n_split, int KB, int skw, int skg) {
  return r4d_gemm_trellis_nt_i8_check(M, K, N, n_split, KB, skw, skg);
}
// SKG * M * N fp32.
inline size_t GemmTrellisNtI8WsBytes(int M, int N, int SKG) { return r4d_gemm_trellis_nt_i8_ws_bytes(M, N, SKG); }

// The coarse-scale sibling (R4DX_PREFILL_INT8_SCALES=coarse; r4d.h "COARSE-scale sibling"): one activation scale per row
// (SA [256] per part, TrellisI8QuantActRow) and one weight scale per column (SWC [N], TrellisI8WscaleColBuild), the same
// GEMM entry and legality (GemmTrellisNtI8Check, GemmTrellisNtI8WsBytes) as the per-128 one.
inline size_t TrellisI8WscaleColCount(int K, int N) { return r4d_trellis_i8_wscale_col_count(K, N); }
inline void TrellisI8WscaleColBuild(const void* w, void* swc, int K, int N, int KB, hipStream_t stream) {
  r4d_trellis_i8_wscale_col(reinterpret_cast<int64_t>(w), reinterpret_cast<int64_t>(swc), K, N, KB,
                            reinterpret_cast<int64_t>(stream));
}
// Diagnostic (tests): TrellisI8DumpW for the coarse table.
inline void TrellisI8DumpWCol(const void* w, const void* swc, void* w8, void* wp, int K, int N, int KB,
                              hipStream_t stream) {
  r4d_trellis_i8_dump_w_col(reinterpret_cast<int64_t>(w), reinterpret_cast<int64_t>(swc), reinterpret_cast<int64_t>(w8),
                            reinterpret_cast<int64_t>(wp), K, N, KB, reinterpret_cast<int64_t>(stream));
}
inline void TrellisI8QuantActRow(const void* x, void* a8, void* sa, int parts, int64_t part_stride, int K,
                                 hipStream_t stream) {
  r4d_trellis_i8_quant_act_row(reinterpret_cast<int64_t>(x), reinterpret_cast<int64_t>(a8),
                               reinterpret_cast<int64_t>(sa), parts, part_stride, K, reinterpret_cast<int64_t>(stream));
}
inline void GemmTrellisNtI8c(const void* a8_0, const void* sa_0, const void* a8_1, const void* sa_1, int n_split,
                             const void* w, const void* swc, const void* svh, void* c, void* ws, void* tickets,
                             int M, int K, int N, int KB, int skw, int skg, float out_scale, hipStream_t stream) {
  r4d_gemm_trellis_nt_i8c(reinterpret_cast<int64_t>(a8_0), reinterpret_cast<int64_t>(sa_0),
                          reinterpret_cast<int64_t>(a8_1), reinterpret_cast<int64_t>(sa_1), n_split,
                          reinterpret_cast<int64_t>(w), reinterpret_cast<int64_t>(swc), reinterpret_cast<int64_t>(svh),
                          reinterpret_cast<int64_t>(c), reinterpret_cast<int64_t>(ws),
                          reinterpret_cast<int64_t>(tickets), M, K, N, KB, skw, skg, out_scale,
                          reinterpret_cast<int64_t>(stream));
}

// R4DX_FAKEQ_W (r4d.h "weight fake-quant", docs/int8-prefill.md): a MEASUREMENT hook. The scale table of one
// trellis linear (TrellisWscaleCount floats, [K / 16 >> gsh][N]) and the three launches of the entries above
// with every decoded weight fragment rounded to symmetric int8 and back by it. `wscale` is the table, `gsh`
// log2 of the k-tiles per scale group (3 = 128 k, 1 = 32 k); NT is ignored (the rounding kernels always load
// weights temporally).
inline size_t TrellisWscaleCount(int K, int N, int gsh) { return r4d_trellis_wscale_count(K, N, gsh); }
inline void TrellisWscaleBuild(const void* w, void* scale, int K, int N, int KB, int gsh,
                               hipStream_t stream) {
  r4d_trellis_wscale_f32(reinterpret_cast<int64_t>(w), reinterpret_cast<int64_t>(scale), K, N, KB, gsh,
                         reinterpret_cast<int64_t>(stream));
}
inline void GemmTrellisNtM64Wq(const void* a0, const void* a1, int n_split, const void* w,
                                const void* svh, void* c, void* ws, void* tickets, int M, int K,
                                int N, int KB, int WV, int SK, int MT, int NP, int SKG, int U, int NT,
                                float out_scale, const void* wscale, int gsh, hipStream_t stream) {
  r4d_gemm_trellis_nt_m64_wq(reinterpret_cast<int64_t>(a0), reinterpret_cast<int64_t>(a1), n_split,
                             reinterpret_cast<int64_t>(w), reinterpret_cast<int64_t>(svh),
                             reinterpret_cast<int64_t>(c), reinterpret_cast<int64_t>(ws),
                             reinterpret_cast<int64_t>(tickets), M, K, N, KB, WV, SK, MT, NP, SKG, U,
                             NT, out_scale, reinterpret_cast<int64_t>(wscale), gsh,
                             reinterpret_cast<int64_t>(stream));
}
// _raw: fp32 C = A @ Q' (no output transform), Q' the rounded Q; one-hot A rows return rows of Q' bit for bit.
inline void GemmTrellisNtM64RawWq(const void* a0, const void* a1, int n_split, const void* w, void* c,
                                   void* ws, void* tickets, int M, int K, int N, int KB, int WV, int SK,
                                   int MT, int NP, int SKG, int U, int NT, const void* wscale, int gsh,
                                   hipStream_t stream) {
  r4d_gemm_trellis_nt_m64_raw_wq(reinterpret_cast<int64_t>(a0), reinterpret_cast<int64_t>(a1), n_split,
                                 reinterpret_cast<int64_t>(w), reinterpret_cast<int64_t>(c),
                                 reinterpret_cast<int64_t>(ws), reinterpret_cast<int64_t>(tickets), M, K,
                                 N, KB, WV, SK, MT, NP, SKG, U, NT, reinterpret_cast<int64_t>(wscale),
                                 gsh, reinterpret_cast<int64_t>(stream));
}
inline void GemmTrellisNtM256Wq(const void* a0, const void* a1, int n_split, const void* w,
                                 const void* svh, void* c, void* ws, void* tickets, int M, int K, int N,
                                 int KB, int SK, int NP, int SKG, int U, float out_scale, int skw,
                                 const void* wscale, int gsh, hipStream_t stream) {
  r4d_gemm_trellis_nt_m256_wq(reinterpret_cast<int64_t>(a0), reinterpret_cast<int64_t>(a1), n_split,
                              reinterpret_cast<int64_t>(w), reinterpret_cast<int64_t>(svh),
                              reinterpret_cast<int64_t>(c), reinterpret_cast<int64_t>(ws),
                              reinterpret_cast<int64_t>(tickets), M, K, N, KB, SK, NP, SKG, U, out_scale,
                              reinterpret_cast<int64_t>(stream), skw, reinterpret_cast<int64_t>(wscale),
                              gsh);
}

inline void DflashConvT2G16Bf16(const void* x, const void* delta, const void* base, void* out,
                                 int T, int H, int dpitch, int NG, int taps, int group,
                                 int block_size, hipStream_t stream) {
  R4DX_R4D_CHECK("dflash_conv_t2_g16_bf16",
                 r4d_dflash_conv_t2_g16_bf16(x, delta, base, out, T, H, dpitch, NG, taps, group,
                                             block_size, stream));
}

// ---- paging helpers ---------------------------------------------------------------------
// Contiguous per-sequence block table: sequence s owns blocks [s*max_blocks, (s+1)*max_blocks),
// laid out row-major [num_seqs, max_blocks] the way R4DArgs::block_table expects.
inline std::vector<int32_t> BuildContiguousBlockTable(int num_seqs, int max_blocks) {
  std::vector<int32_t> table(static_cast<size_t>(num_seqs) * max_blocks);
  for (int s = 0; s < num_seqs; ++s) {
    for (int b = 0; b < max_blocks; ++b) {
      table[static_cast<size_t>(s) * max_blocks + b] = s * max_blocks + b;
    }
  }
  return table;
}

// ---- registry -----------------------------------------------------------------------------
inline int KernelCount() { return r4d_kernel_count(); }
inline const R4DKernelInfo* KernelAt(int i) { return r4d_kernel_at(i); }

}  // namespace r4dx::core::r4d
