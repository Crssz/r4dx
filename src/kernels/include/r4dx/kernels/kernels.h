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

// ---- activation-quant epilogue selector (docs/r9700.md R2/P2, 2026-09-20) ---------------------
// Selects an OPTIONAL fused quantization epilogue, applied by a producer kernel to a row it has
// ALREADY fully computed and written as bf16 (rmsnorm's own `out`, residual_rmsnorm's `out_normed`,
// silu_mul's `out`), producing exactly the bytes+scale the corresponding standalone quant kernel
// would have produced from those same bf16 values -- byte-identical by construction, since the
// epilogue re-reads the just-written bf16 row and runs the IDENTICAL reduction+quantize algorithm
// as the kernel it stands in for (verified byte-exact by tests/kernels/test_fused_quant.cpp before
// any call site was wired to use this):
//   r4dx_epilogue_f16          vs r4dx_model_cast_bf16_to_f16 (src/model/kernels/model_kernels.hip)
//   r4dx_epilogue_fp8_e4m3_row vs r4dx_quant_act_fp8e4m3_row (this header, below)
//   r4dx_epilogue_int8_fraga8  vs third_party/libr4d's r4d_quant_act_i8 (same per-row
//                              scale=max(1e-8,absmax)/127 and WMMA-fragment byte permute: dest
//                              byte i of a 16-byte k-step s=i>>4 reads source column
//                              src = (s<<4) + 8*((j&7)>>2) + 4*(j>>3) + (j&3), j=i&15)
// r4dx_epilogue_none (0, the default on every entry point below) reproduces the exact pre-this-
// pass behavior: no epilogue computed, `epilogue_out`/`epilogue_scale` ignored (may be 0/nullptr).
// This is a genuine SEPARATE reduction pass over the row (re-reading the bf16 output the kernel
// just wrote), not a single-pass fusion that would change the row's own bf16 rounding -- it exists
// to fold what would otherwise be a SEPARATE kernel LAUNCH (and the standalone kernel's own read
// of this same bf16 row from global memory, which the epilogue pays for here instead) into this
// launch; see docs/status.md's "R2/P2" section for the measured launch-count effect. All three
// epilogue kinds run as one uniform extra pass AFTER `out`/`out_normed` is fully written (re-reading
// it from global memory, same as the standalone kernel they replace would have) -- f16 needs no
// row reduction so this pass is a plain per-element convert; fp8/int8 additionally compute a
// per-row absmax reduction first. epilogue_scale is unused (may be 0) for r4dx_epilogue_f16; for
// the two per-row-scaled formats it is a [rows] fp32 device buffer, one scale written per row.
enum r4dx_epilogue {
  r4dx_epilogue_none = 0,
  r4dx_epilogue_f16 = 1,
  r4dx_epilogue_fp8_e4m3_row = 2,
  r4dx_epilogue_int8_fraga8 = 3,
};

extern "C" {

// ---- rmsnorm --------------------------------------------------------------------------------
// out[row,:] = (x[row,:] * rsqrt(mean(x[row,:]^2) + eps)) * (1 + weight[:])   (Qwen3_5RMSNorm's
// zero-centered weight convention -- the stored weight is w such that the effective scale is
// 1+w, so an all-zero weight is the identity). fp32 accumulation, bf16 in/out. x: [rows, hidden]
// bf16, row-major. weight: [hidden] bf16.
// `epilogue`/`epilogue_out`/`epilogue_scale`: see the r4dx_epilogue doc above. epilogue_out is
// [rows,hidden] in the selected format (f16 uint16_t*, fp8 uint8_t*, or int8 int8_t*);
// epilogue_int8_fraga8 requires hidden % 16 == 0 (matches r4d_quant_act_i8's own precondition).
void r4dx_rmsnorm_bf16(int64_t x, int64_t weight, int64_t out, int64_t rows, int64_t hidden,
                        float eps, int64_t stream, int epilogue = r4dx_epilogue_none,
                        int64_t epilogue_out = 0, int64_t epilogue_scale = 0);

// Fused residual-add + rmsnorm: sum = x + residual (fp32 accumulate); out_residual = bf16(sum)
// (the new residual carried into the next block); out_normed = rmsnorm(sum) * (1+weight). Saves
// one read/write of [rows,hidden] versus calling r4dx_residual_add_bf16 then r4dx_rmsnorm_bf16.
// `epilogue`/`epilogue_out`/`epilogue_scale`: applied to `out_normed` only (the value a caller
// feeds into a quantized GEMM next -- `out_residual` never is), see r4dx_epilogue doc above.
void r4dx_residual_rmsnorm_bf16(int64_t x, int64_t residual, int64_t weight,
                                 int64_t out_residual, int64_t out_normed, int64_t rows,
                                 int64_t hidden, float eps, int64_t stream,
                                 int epilogue = r4dx_epilogue_none, int64_t epilogue_out = 0,
                                 int64_t epilogue_scale = 0);

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
// `epilogue`/`epilogue_out`/`epilogue_scale`: applied to `out`, see r4dx_epilogue doc above.
void r4dx_silu_mul_bf16(int64_t gate_up, int64_t out, int64_t rows, int64_t intermediate,
                         int64_t in_row_stride, int64_t stream,
                         int epilogue = r4dx_epilogue_none, int64_t epilogue_out = 0,
                         int64_t epilogue_scale = 0);

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
// point of this entry point over the host gather. Bounds-checked (review finding, 2026-09-20,
// fixed): an id outside [0, vocab) clamps to row 0 in-kernel instead of reading out-of-bounds
// device memory -- unlike r4dx::kernels::EmbeddingGatherHost, which throws std::out_of_range for
// the same condition, this path degrades to a wrong-but-safe embedding row rather than an
// exception (a device fault mid-kernel would otherwise be unrecoverable, and this entry point's
// callers include Model::RunChunk/VerifyWindow's ordinary caller-supplied prompt tokens whenever
// ModelOptions::embed_device_resident is true, not just MtpHead::Draft's own argmax-fed ids, which
// were already safe by construction). table: [vocab, hidden] bf16 device pointer. ids: [n] int32
// device pointer. out: [n, hidden] bf16 device pointer. vocab: table's row count, for the guard.
void r4dx_embedding_gather_bf16(int64_t table, int64_t ids, int64_t out, int64_t n, int64_t hidden,
                                 int64_t vocab, int64_t stream);

// ---- device-side single/few-element gather (docs/r9700.md R9, reduced-vocab MTP draft head) ----
// out[i] = table[idx[i]] for i in [0, n), entirely on-device -- the general-purpose counterpart of
// r4dx_embedding_gather_bf16 above but for a plain int32 lookup table instead of a [vocab,hidden]
// embedding row table. Built for MtpHead::Draft's reduced-vocab draft head (docs/mtp.md "reduced-
// vocab draft head"): after computing argmax over the SUBSET logits (r4dx_argmax_f32 with
// vocab==draft_vocab_size), the result is a subset-local index, not a real vocabulary id -- this
// kernel maps it back to the real id via the container's `mtp.draft_head.vocab_ids` table with zero
// host round-trips, so the chained device-resident draft loop (idx feeds embedding gather feeds the
// next draft step) never needs to leave the device. `n` is always small (1 in every current caller)
// but this is not hardcoded to n==1 in case a future caller wants to gather a few indices in one
// launch. Bounds-checked the same way r4dx_embedding_gather_bf16 is: an idx outside [0, table_size)
// clamps to table[0] rather than reading out-of-bounds device memory (same "degrade to a
// wrong-but-safe value, never fault mid-kernel" reasoning as that kernel's own doc comment).
// table: [table_size] int32 device pointer. idx: [n] int32 device pointer (subset-local indices).
// out: [n] int32 device pointer (may alias idx's own buffer only if n==1 -- see call sites).
void r4dx_gather_i32(int64_t table, int64_t idx, int64_t out, int64_t n, int64_t table_size,
                      int64_t stream);

// ==== DFlash2 drafter device pieces (docs/dflash2.md "Kernels", Milestone 5 S1) ================
// The five device-side primitives the DFlash2 draft forward needs that nothing in this repo or in
// third_party/libr4d already provides. Every convention below (pairing, theta, norm form, SWA
// visibility rule, conv delta/base layout, tie-break) is docs/dflash2.md's, not a guess; each one
// has a ctest unit test under tests/kernels/ that checks it against a CPU reference AND against
// the Python reference's fixture A (tools/reference/golden_out/dflash2/fixture_a).
//
// NOTE on shapes: these are purpose-built for the DFlash2 draft block, not general-purpose. Each
// entry point states its preconditions and the host wrapper throws std::runtime_error (not a
// silent wrong answer) when one is violated.

// ---- rope: full-width NeoX split-half, per-row absolute positions -----------------------------
// In-place rotation of ALL `head_dim` dims of each head (docs/dflash2.md section 3: n_rot == 128 ==
// head_dim, so unlike r4dx_rope_partial_mrope_bf16 above there is no pass-through tail), NeoX
// split-half pairing (element i pairs with i + head_dim/2), theta 1e7 for the real drafter. fp32
// math, RTNE back to bf16.
//
//   half = head_dim/2;  angle[i] = pos * theta^(-2i/head_dim),  i in [0, half)
//   out[i]      = x[i]*cos - x[i+half]*sin
//   out[i+half] = x[i+half]*cos + x[i]*sin
//
// This is mathematically the `rotary_dim == head_dim` case of r4dx_rope_partial_mrope_bf16, and is
// a separate entry point for two reasons: (a) the DFlash2 INJECTION path ropes K only (there is no
// Wq in that path at all -- docs/dflash2.md section 4.1), which that kernel cannot express since it
// unconditionally dereferences both q and k; (b) `theta`/`head_dim`/no-tail is this drafter's
// contract, and mixing it into the target model's partial-rotary entry point would make a future
// change to either silently affect the other.
//
// q: [rows, heads_q, head_dim] bf16, in place -- pass q=0 AND heads_q=0 to rope K only.
// k: [rows, heads_k, head_dim] bf16, in place -- pass k=0 AND heads_k=0 to rope Q only.
// pos_ids: [rows] int32 DEVICE pointer, the tokens' absolute sequence positions.
// Precondition: head_dim even.
void r4dx_rope_neox_bf16(int64_t q, int64_t k, int64_t pos_ids, int rows, int heads_q, int heads_k,
                          int head_dim, float theta, int64_t stream);

// ---- per-row top-16 -----------------------------------------------------------------------
// out_ids[r, 0..15] / out_vals[r, 0..15] = the 16 largest entries of logits[r, 0..vocab), sorted
// DESCENDING by value, ties broken deterministically toward the LOWER id (the total order is
// "(v, i) beats (v', i') iff v > v' || (v == v' && i < i')"). This is `ggml_top_k`'s contract as
// the DFlash2 selector consumes it (docs/dflash2.md section 4.2: `cand, unary = top16(logits)`).
//
// One workgroup per row; each thread keeps a register-resident sorted top-16 over its own
// grid-strided slice, then the 256 per-thread lists are merged pairwise in LDS (8 rounds). The
// merge uses the same total order, so the result is exactly the global top-16 under it --
// deterministic run to run, independent of thread scheduling.
//
// logits: [rows, vocab] fp32 device pointer, rows contiguous (row stride == vocab).
// out_ids: [rows, 16] int32. out_vals: [rows, 16] fp32.
// Preconditions: 1 <= rows <= 8 (the DFlash2 block size; the kernel is correct for any rows but
// the launch is sized for a handful of rows, see the wrapper), vocab >= 16.
void r4dx_topk16_f32(int64_t logits, int64_t out_ids, int64_t out_vals, int rows, int64_t vocab,
                      int64_t stream);

// ---- DFlash2 draft-block attention: non-causal, windowed, GQA ---------------------------------
// The draft block's own attention (docs/dflash2.md section 4.2 / the "SWA visibility rule" row of
// section 2's table). For query row t (absolute position q_pos = n_injected + t) the visible key
// set is:
//   * injected-store positions p in [max(store_begin, q_pos - window + 1), n_injected - 1]   (the
//     SWA rule `q_pos - p < window`, plus "only already-injected positions"), read at slot
//     p % slots;
//   * ALL T of the block's own keys, unconditionally -- the block is non-causal, and a block key
//     in the query's future is never masked (`attention.causal=false`; the SWA rule's
//     `q_pos - p < window` is trivially true for a negative difference).
// Softmax in fp32 (numerically stable two-pass: the visible row's scores are materialised in LDS,
// then max-subtracted, exponentiated and normalised there), bf16 output. GQA: q-head h reads
// kv-head h / (heads_q/heads_kv).
//
// q:        [T, heads_q,  head_dim] bf16 -- post-q_norm, post-rope.
// k_block:  [T, heads_kv, head_dim] bf16 -- post-k_norm, post-rope. SCRATCH: never written to the
// v_block:  [T, heads_kv, head_dim] bf16    store by this kernel (docs/dflash2.md section 5: the
//                                           block's own K/V is discarded every round, which is why
//                                           the ring needs no rollback).
// k_store:  [slots, heads_kv, head_dim] bf16 -- the injected-feature ring, slot = position % slots.
// v_store:  [slots, heads_kv, head_dim] bf16
// out:      [T, heads_q, head_dim] bf16.
// n_injected: number of positions already injected (== the block's start position).
// store_begin: the FIRST position in the ring whose contents are valid, i.e. the visible store is
//   the CONTIGUOUS run [store_begin, n_injected) intersected with the sliding window. 0 (the
//   original behaviour, and the only value any caller passed before) means "everything injected so
//   far is valid". A caller that stopped injecting for a while and then resumed at a higher
//   position (r4dx::model::DflashDraft's cold-ring gap, docs/dflash2.md section 5) passes the
//   resume position here, and the ring bytes for the skipped positions are then never read -- which
//   is why they need not be cleared, exactly the "self-correcting via position overwrite" argument
//   the append-only case already relies on. Slot mapping stays `p % slots` over TRUE absolute
//   positions either way, so nothing about rope or the ring geometry changes.
// Preconditions (throw, not silently wrong): 1 <= T <= 8; head_dim <= 128 and head_dim % 32 == 0;
// heads_q % heads_kv == 0 and the ratio <= 4; 1 <= window <= 2048; window <= slots (so the visible
// store range can never alias itself in the ring); 0 <= store_begin <= n_injected.
void r4dx_dflash_attn_bf16(int64_t q, int64_t k_block, int64_t v_block, int64_t k_store,
                            int64_t v_store, int64_t out, int T, int heads_q, int heads_kv,
                            int head_dim, int n_injected, int store_begin, int window, int slots,
                            float scale, int64_t stream);

// ---- DFlash2 grouped dynamic depthwise conv (thin wrapper over libr4d) -------------------------
// out[t,c] = (base[side,0,c] + dyn[t, side*taps*NG + 0*NG + g]) * x[t,c]
//          + (base[side,1,c] + dyn[t, side*taps*NG + 1*NG + g]) * x[t-1,c] * (t >= 1),  g = c/16
// with taps = 2 and group = 16 (the only geometry libr4d compiles: r4d_dflash_conv_t2_g16_bf16).
// This is NOT a new r4dx kernel -- it is the address arithmetic that turns the container's own
// tensor layouts into that entry point's (x, delta, base, dpitch, NG) contract, which is the part
// that is easy to get wrong and is what the unit test pins:
//   delta = dyn + side*taps*NG        (the [T, 2, taps, NG] projection's side-major slice;
//                                      dpitch stays the FULL 2*taps*NG row pitch, per r4d.h:118)
//   base  = base + side*taps*H        (base is [2(side)][taps][H], channel fastest)
//   NG    = H / group
// `block_size` must be a power of two >= T; the caller passes ONE block starting at row 0, so
// r4d_dflash_conv_body's `(t & blockmask) >= tap` degenerates to `t >= tap` (docs/dflash2.md's
// "Shift-by-one / block-start masking" row). `out` must not alias `x` (libr4d's own precondition).
//
// x: [T, H] bf16. dyn: [T, 2*taps*NG] bf16. base: [2, taps, H] bf16. out: [T, H] bf16.
// Deliberately does NOT increment r4dx_kernel_launch_counter: that counter's documented contract
// (below) is "r4dx-owned launches only, never third_party/libr4d's", and the launch this makes is
// libr4d's.
void r4dx_dflash_conv_bf16(int64_t x, int64_t dyn, int64_t base, int64_t out, int T, int H,
                            int side, int block_size, int64_t stream);

// ---- plain (non zero-centered) rmsnorm ---------------------------------------------------------
// out[row,:] = x[row,:] * rsqrt(mean(x[row,:]^2) + eps) * weight[:]
// i.e. the PLAIN weight form, NOT r4dx_rmsnorm_bf16's `(1 + weight)` Qwen3_5RMSNorm convention at
// the top of this header. DFlash2 is a llama.cpp-native checkpoint whose every norm site is ggml's
// plain `ggml_rms_norm` + `ggml_mul` (docs/dflash2.md's "RMSNorm convention" row); feeding a
// DFlash2 norm weight to r4dx_rmsnorm_bf16 would add a spurious +1 to every channel. Nothing in
// this repo or in third_party/libr4d computed this form before (r4d_gdn_gated_rmsnorm_h128_bf16 is
// the gated per-head variant, a different op), hence a new entry point rather than a flag.
// fp32 accumulation. x: [rows, hidden] bf16. weight: [hidden] bf16.
// out: [rows, hidden], bf16 when out_fp32 == 0, fp32 when out_fp32 != 0 (the encoder path wants
// fp32 to feed a host-side check / a downstream fp32 consumer; the layer path wants bf16).
// In-place (out == x, out_fp32 == 0) is supported -- see the kernel's own comment.
void r4dx_rmsnorm_plain_bf16(int64_t x, int64_t weight, int64_t out, int64_t rows, int64_t hidden,
                              float eps, int out_fp32, int64_t stream);

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
