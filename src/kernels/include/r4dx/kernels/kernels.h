// r4dx-owned HIP kernels: the glue r4d.h does not provide (docs/architecture.md "Own kernels").
// Plain C ABI, same convention as r4d.h itself: device pointers as int64_t, a stream as int64_t
// (0 = the default/legacy stream), strides and shapes in ELEMENTS. Implemented in
// src/kernels/src/r4dx_kernels.hip, compiled by hipcc the same way third_party/CMakeLists.txt
// compiles r4d_core (CMake's HIP language fights clang-cl on this toolchain -- see
// docs/build-windows.md), and linked into the r4dx_kernels static library.
//
// Dtypes, spelled out per entry point below:
//   bf16   -- 16-bit brain float, r4dx_core::FloatToBf16/Bf16ToFloat round-trip it on the host.
//   fp8e4m3 -- OCP E4M3FN, r4dx_core::FloatToFp8E4M3/Fp8E4M3ToFloat.
//   All shapes/strides are in ELEMENTS of the tensor's own dtype, never bytes.
#pragma once

#include <cstdint>

extern "C" {

// ---- rmsnorm --------------------------------------------------------------------------------
// out[row,:] = (x[row,:] * rsqrt(mean(x[row,:]^2) + eps)) * (1 + weight[:])   (Qwen3_5RMSNorm's
// zero-centered weight convention -- the stored weight is w such that the effective scale is
// 1+w, so an all-zero weight is the identity). fp32 accumulation, bf16 in/out. x: [rows, hidden]
// bf16, row-major. weight: [hidden] bf16.
void r4dx_rmsnorm_bf16(int64_t x, int64_t weight, int64_t out, int64_t rows, int64_t hidden,
                        float eps, int64_t stream);

// Fused residual-add + rmsnorm: sum = x + residual (fp32 accumulate); out_residual = bf16(sum)
// (the new residual carried into the next block); out_normed = rmsnorm(sum) * (1+weight). Saves
// one read/write of [rows,hidden] versus calling r4dx_residual_add_bf16 then r4dx_rmsnorm_bf16.
void r4dx_residual_rmsnorm_bf16(int64_t x, int64_t residual, int64_t weight,
                                 int64_t out_residual, int64_t out_normed, int64_t rows,
                                 int64_t hidden, float eps, int64_t stream);

// ---- residual add -----------------------------------------------------------------------------
// out[i] = bf16(fp32(a[i]) + fp32(b[i])), elementwise over `n` elements.
void r4dx_residual_add_bf16(int64_t a, int64_t b, int64_t out, int64_t n, int64_t stream);

// ---- silu_mul ---------------------------------------------------------------------------------
// One row per workgroup, consuming the FUSED mlp.gate_up projection layout docs/container-format.md
// mandates: within row r, gate is gate_up[r*in_row_stride + 0 .. intermediate) and up is
// gate_up[r*in_row_stride + intermediate .. 2*intermediate). out[r,i] = silu(gate[r,i]) * up[r,i],
// silu(x) = x * sigmoid(x), fp32 compute. `in_row_stride` is ordinarily 2*intermediate but kept
// explicit in case a caller's gate_up buffer has row padding.
// gate_up: [rows, in_row_stride] bf16 (>= 2*intermediate valid columns per row). out: [rows,
// intermediate] bf16, row stride == intermediate (contiguous).
// NOTE: a flat two-pointer (gate[], up[], n) form was tried first and dropped -- it has no row
// concept, so for rows>1 (any chunked-prefill call) the natural gate=C/up=C+intermediate call a
// src/model author would write from the fused layout silently read across row boundaries.
void r4dx_silu_mul_bf16(int64_t gate_up, int64_t out, int64_t rows, int64_t intermediate,
                         int64_t in_row_stride, int64_t stream);

// ---- rope: partial rotary, text-only mrope ---------------------------------------------------
// Rotates the first `rotary_dim` (64 = head_dim * partial_rotary_factor 0.25) dims of each head
// in place, NeoX/half-split pairing (rotate_half: element i pairs with i + rotary_dim/2), matching
// modeling_qwen3_5.py's apply_rotary_pos_emb (`transformers.models.glm.modular_glm`-derived, NOT
// GPT-J interleaved-pair rope). theta = 1e7. Dims [rotary_dim, head_dim) pass through unchanged.
//
// mrope_interleaved (config) describes how the THREE position streams (temporal, height, width)
// interleave across the 32 rotary frequency bins when a prompt has image/video tokens -- index i
// mod 3 selects which stream's position id feeds bin i (mrope_section=[11,11,10]), NOT the
// even/odd pairing of a rotated element with its partner (that pairing is NeoX half-split
// regardless of mrope). This entry point implements the TEXT-ONLY case, where all three streams
// equal the plain token position, so the three-stream selection is a no-op and this reduces to
// ordinary 1D rope over `pos_ids`.
// TODO(vision milestone): a multimodal caller needs three per-token position ids (t,h,w) instead
// of one; add r4dx_rope_partial_mrope_multimodal_bf16 (or a `pos_ids` shaped [3, tokens]) then,
// selecting stream `i % 3` per frequency bin `i` exactly as
// Qwen3_5TextRotaryEmbedding.recomposition_frequencies does. Not implemented here.
//
// q: [tokens, heads_q, head_dim] bf16, in place. k: [tokens, heads_k, head_dim] bf16, in place.
// pos_ids: [tokens] int32.
void r4dx_rope_partial_mrope_bf16(int64_t q, int64_t k, int64_t pos_ids, int tokens, int heads_q,
                                   int heads_k, int head_dim, int rotary_dim, float theta,
                                   int64_t stream);

// ---- fp8 e4m3 activation quantisation (feeds r4d_gemm_mxfp4a8_nt_m64) ------------------------
// Per-row: scale = max(1e-8, max_k |x[row,k]|) / 448 (448 = e4m3fn's max finite magnitude);
// q[row,k] = fp8e4m3(x[row,k] / scale), plain row-major byte order (r4d_gemm_mxfp4a8_nt_m64 reads
// the A operand as a flat [M,K] byte array -- unlike r4d_quant_act_i8's int8 path, there is no
// WMMA-fragment byte reorder on the fp8 activation side; verified against
// third_party/libr4d/r4d_gemm_mxfp4a8_nt_m64.hip's `A + r*K + kg + mhalf` addressing and
// test_mxfp4_gemm.py's plain `af8 = a.to(torch.float8_e4m3fn)`). x: [M,K] bf16. q: [M,K] fp8e4m3
// (uint8). scale: [M] fp32.
// Precondition (matches r4d_gemm_mxfp4a8_nt_m64's own A-operand addressing): K must be a multiple
// of 16, since that kernel reads the A operand as 2-dword (8-byte) chunks at byte offset
// `r*K + kg + mhalf`; see tests/kernels/test_mxfp4_gemm.cpp.
void r4dx_quant_act_fp8e4m3_row(int64_t x, int64_t q, int64_t scale, int M, int K,
                                 int64_t stream);

// ---- paged fp8 KV cache write -------------------------------------------------------------
// Writes T new (k,v) rows into the paged fp8 e4m3 HND K/V-interleaved cache exactly as
// R4DArgs.kv expects (r4d.h:44-58, docs/architecture.md "fp8 KV paging"): cache shape
// (num_blocks, kv_heads, block_size, 2*head_dim), K at column [0,head_dim), V at
// [head_dim,2*head_dim) per slot. slot_mapping[t] = block_id*block_size + offset (the caller's
// own paging decision); kv_block_stride/kv_head_stride are R4DArgs' own element strides so this
// writes into the identical buffer the attention kernels read. Quantises with the per-(layer,
// head) STATIC descale already resident in the container (k_descale/v_descale, placeholder 1.0
// until calibration): stored_fp8 = fp8e4m3(real_bf16_value / descale[head]), the inverse of what
// the attention kernel's own dequant (value * descale) expects to recover.
//
// slot_mapping[t] == -1 means "skip this token" (the universal batching-layer convention for a
// padded row); the kernel returns for that (t,h) pair without touching the cache. Any other
// negative value is undefined -- callers must use exactly -1 to skip, never another sentinel.
//
// k_new,v_new: [T, kv_heads, head_dim] bf16. slot_mapping: [T] int32 (>= 0, or exactly -1 to
// skip). k_descale,v_descale: [kv_heads] fp32 -- this is the WRITER's own per-(layer,head) static
// descale table, one row, NOT R4DArgs.k_descale/v_descale (which r4d_attn_decode_h256_gqa6.hip
// indexes as `seq * kv_heads + kvh`, i.e. a [num_seqs, kv_heads] table -- a caller feeding
// R4DArgs must broadcast/replicate this same [kv_heads] row num_seqs times; see
// r4dx::core::r4d::AttnDecodeFp8Kv's comment in r4d.hpp). kv_cache: (num_blocks, kv_heads,
// block_size, 2*head_dim) fp8e4m3 (uint8), R4DArgs.kv-compatible.
void r4dx_kv_write_paged_fp8_hnd(int64_t k_new, int64_t v_new, int64_t slot_mapping,
                                  int64_t k_descale, int64_t v_descale, int64_t kv_cache,
                                  int T, int kv_heads, int head_dim, int block_size,
                                  int64_t kv_block_stride, int64_t kv_head_stride,
                                  int64_t stream);

// ---- device argmax (host-overhead pass, 2026-09-19) -------------------------------------------
// out_idx[0] = argmax_i logits[i] (ties broken toward the lowest index, matching
// r4dx::kernels::Argmax's CPU reference in sampler.hpp). One block only -- vocab (~250k floats,
// ~1MB) comfortably fits one block's grid-stride loop, and this exists specifically so a
// temperature==0 (greedy) decode caller can skip the vocab-sized logits D2H copy entirely and
// read back a single int32 instead (Model::DecodeStepGreedy, model.cpp). logits: [vocab] fp32.
// out_idx: device int32[1].
void r4dx_argmax_f32(int64_t logits, int64_t out_idx, int64_t vocab, int64_t stream);

// ---- device-resident embedding gather (MTP device-residency pass, docs/mtp.md) -----------------
// out[row,:] = table[ids[row],:], entirely on-device -- the device-resident counterpart of
// r4dx::kernels::EmbeddingGatherHost (embedding.hpp) for a text.embed_tokens table that has been
// uploaded to VRAM (Container::EmbedTokensDevice()). `ids` is a DEVICE int32 pointer, not host --
// in particular r4dx_argmax_f32's own `out_idx` output can feed straight into this with zero host
// syncs in between (MtpHead::Draft's chained per-draft-token loop, model.cpp), which is the whole
// point of this entry point over the host gather. No bounds check on ids (trusts the caller, same
// convention as every other kernel in this header) -- an id outside [0, vocab) reads out-of-bounds
// device memory. table: [vocab, hidden] bf16 device pointer. ids: [n] int32 device pointer. out:
// [n, hidden] bf16 device pointer.
void r4dx_embedding_gather_bf16(int64_t table, int64_t ids, int64_t out, int64_t n, int64_t hidden,
                                 int64_t stream);

// ---- kernel launch counter (docs/r9700.md P2/task item 4, 2026-09-20) -------------------------
// A plain process-global counter (not thread-safe by design -- Model is single-worker-thread per
// model.h's own SCOPE comment, so this needs no atomic/lock any more than PickTuning's cache does)
// incremented once per r4dx-owned kernel launch above (every r4dx_* entry point in this header,
// AFTER its `rows/M/T <= 0` early-return check, so a no-op call does not count). Counts ONLY
// r4dx-owned launches -- NOT the r4d_gemm_*/r4d_gdn_*/r4d_attn_* launches in third_party/libr4d,
// which this project does not instrument (out of scope: a third_party submodule). Used by
// Model::DecodeStepProfiled (model.cpp) to report "r4dx-owned kernel launches per token" before/
// after the R3/P2 fusion pass -- see docs/mtp.md and docs/status.md for the measured before/after.
void r4dx_kernel_launch_counter_reset();
int64_t r4dx_kernel_launch_counter_get();

}  // extern "C"
