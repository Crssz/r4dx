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
inline void AttnPrefillBf16Kv(const Args& a, hipStream_t stream) {
  R4DX_R4D_CHECK("attn_prefill_h256_gqa6_bf16kv", r4d_attn_prefill_h256_gqa6_bf16kv(&a, stream));
}
inline void AttnDecodeFp8Kv(const Args& a, hipStream_t stream) {
  R4DX_R4D_CHECK("attn_decode_h256_gqa6_fp8kv", r4d_attn_decode_h256_gqa6_fp8kv(&a, stream));
}
inline void AttnDecodeBf16Kv(const Args& a, hipStream_t stream) {
  R4DX_R4D_CHECK("attn_decode_h256_gqa6_bf16kv", r4d_attn_decode_h256_gqa6_bf16kv(&a, stream));
}
// Bytes of split-KV decode scratch this shape needs; allocate and set Args::scratch before
// calling AttnDecode*.
inline int64_t AttnDecodeScratchBytes(const Args& a) {
  return r4d_attn_decode_h256_gqa6_scratch_bytes(&a);
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
inline void GemmBf16NtM16(const void* a, const void* w, void* c, int M, int K, int N, int WV,
                           int SK, hipStream_t stream) {
  r4d_gemm_bf16_nt_m16(reinterpret_cast<int64_t>(a), reinterpret_cast<int64_t>(w),
                        reinterpret_cast<int64_t>(c), M, K, N, WV, SK,
                        reinterpret_cast<int64_t>(stream));
}
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
inline void GemmW4a8NtM64(const void* a, const void* ascale, const void* wq, const void* ws,
                           void* c, int M, int K, int N, int WV, int SK, int MB, int NPW, int NT,
                           hipStream_t stream) {
  r4d_gemm_w4a8_nt_m64(reinterpret_cast<int64_t>(a), reinterpret_cast<int64_t>(ascale),
                        reinterpret_cast<int64_t>(wq), reinterpret_cast<int64_t>(ws),
                        reinterpret_cast<int64_t>(c), M, K, N, WV, SK, MB, NPW, NT,
                        reinterpret_cast<int64_t>(stream));
}
inline void GemmMxfp4a8NtM64(const void* a, const void* ascale, const void* wq, const void* ws,
                              const void* wref, void* c, int M, int K, int N, int WV, int SK,
                              int MB, int NPW, hipStream_t stream) {
  r4d_gemm_mxfp4a8_nt_m64(reinterpret_cast<int64_t>(a), reinterpret_cast<int64_t>(ascale),
                          reinterpret_cast<int64_t>(wq), reinterpret_cast<int64_t>(ws),
                          reinterpret_cast<int64_t>(wref), reinterpret_cast<int64_t>(c), M, K, N,
                          WV, SK, MB, NPW, reinterpret_cast<int64_t>(stream));
}
inline void QuantActI8(const void* a, void* q, void* s, int M, int K, hipStream_t stream) {
  r4d_quant_act_i8(reinterpret_cast<int64_t>(a), reinterpret_cast<int64_t>(q),
                    reinterpret_cast<int64_t>(s), M, K, reinterpret_cast<int64_t>(stream));
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
