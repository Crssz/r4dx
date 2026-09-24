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
// The multimodal case is r4dx_rope_partial_mrope3_bf16 below.
//
// q: [tokens, heads_q, head_dim] bf16, in place. k: [tokens, heads_k, head_dim] bf16, in place.
// pos_ids: [tokens] int32.
void r4dx_rope_partial_mrope_bf16(int64_t q, int64_t k, int64_t pos_ids, int tokens, int heads_q,
                                   int heads_k, int head_dim, int rotary_dim, float theta,
                                   int64_t stream);

// ---- rope: partial rotary, 3-axis (t,h,w) mrope ----------------------------------------------
// The multimodal counterpart of r4dx_rope_partial_mrope_bf16: same rotation, same expression
// order, same NeoX half-split pairing -- the only difference is that each frequency bin draws its
// position from one of THREE per-token position streams instead of a single one.
//
// The bin -> stream assignment is Qwen3_5TextRotaryEmbedding.recomposition_frequencies (read from
// modeling_qwen3_5.py, not inferred): the table starts all-temporal, then the height stream
// overwrites bins `range(1, 3*mrope_section[1], 3)` and the width stream bins
// `range(2, 3*mrope_section[2], 3)`. With this model's mrope_section [11,11,10] and rotary_dim 64
// (32 bins) that happens to reduce to bin index mod 3, but the section bounds are real parameters
// here rather than an assumption -- `sec_*` must be non-negative and sum to rotary_dim/2 (throws
// otherwise), and `sec_t` is validated rather than used (it is the complement of the other two).
//
// Calling this with three IDENTICAL position rows is BIT-IDENTICAL to
// r4dx_rope_partial_mrope_bf16 over that one row (same inv_freq expression, same sincosf, same
// rounding order) -- asserted by tests/kernels/test_rope_mrope3.cpp, and the property that lets a
// caller with a text-only prompt keep taking the cheaper single-row entry point with no numeric
// consequence either way.
//
// pos_ids3: [3, tokens] int32, contiguous, row 0 = temporal, row 1 = height, row 2 = width.
void r4dx_rope_partial_mrope3_bf16(int64_t q, int64_t k, int64_t pos_ids3, int tokens, int heads_q,
                                    int heads_k, int head_dim, int rotary_dim, float theta,
                                    int sec_t, int sec_h, int sec_w, int64_t stream);

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

// The same kernel, also writing the winning VALUE: out_idx[0] = the lowest index among equal maxima
// of logits[0, vocab), out_val[0] = logits[out_idx[0]]. The tensor-parallel greedy merge's per-shard
// half (docs/tp.md 7.3): each rank argmaxes its own lm_head vocab shard, and the host picks the rank
// with the strictly larger value (tie -> the lower rank, i.e. the lower global id). out_val: device
// float[1]. r4dx_argmax_f32 is this with no value output, so TP=1 is unchanged.
void r4dx_argmax_val_f32(int64_t logits, int64_t out_idx, int64_t out_val, int64_t vocab,
                          int64_t stream);

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

// ---- per-row top-K + logsumexp row summary (Milestone 6 S1, docs/sampling.md) ------------------
// The device-side ROW SUMMARY a sampling decode step needs instead of the full [rows, vocab] fp32
// logits D2H: per row, the K = R4DX_TOPK_LSE_K largest RAW logits with their ids (the canonical
// order docs/sampling.md defines: value DESCENDING, ties broken toward the LOWER id -- exactly
// r4dx_topk16_f32's total order, at K = 64 instead of 16), plus that row's logsumexp of the
// TEMPERATURE-SCALED logits. D2H per round is rows * (K*8 + 4) bytes (K=64: 512 + 4 = 516 per row)
// instead of rows * vocab * 4 (~993 KB per row at this model's vocab).
//
//   out_vals[r, 0..K-1] / out_ids[r, 0..K-1] : the top-K of logits[r, 0..vocab) under the total
//     order "(v,i) beats (v',i') iff v > v' || (v == v' && i < i')", sorted descending. RAW logit
//     values, NOT scaled by inv_temperature -- the host applies its own `logits/temperature` to
//     them so the summary path and the full-vocab path (r4dx::kernels::SampleCanonical,
//     sampler.hpp) compute bit-identical softmax numerators over the top-K.
//   out_lse[r] = log( sum_i exp(logits[r,i] * inv_temperature) ), accumulated stably: the row max
//     m is taken from the top-1 above, terms are expf((logits[r,i] - m) * inv_temperature) in fp32
//     and accumulated in DOUBLE, and the result is m*inv_temperature + log(sum). A row whose max
//     is -inf (every entry -inf) yields -inf, not NaN. Measured |error| vs an fp64 CPU reference
//     is <= 1e-4 over the inputs tests/kernels/test_topk_lse.cpp covers (see docs/sampling.md).
//
// Algorithm (why it is exactly the top-64 and not an approximation): the row is split across up to
// 64 workgroups, each owning a contiguous slice. A block keeps a per-thread register-resident
// top-16 and merges the 256 lists in LDS exactly as r4dx_topk16_f32 does, then REPEATS that
// extraction four times, each round admitting only elements strictly WORSE than the previous
// round's last emitted (value, id) pair under the same total order -- so round r yields exactly
// ranks 16r+1 .. 16r+16 of the slice. A second kernel merges the per-slice top-64s the same way
// (at most 64 of the row's top-64 can lie in one slice, so that slice's own top-64 holds all of
// them) and combines the per-slice (max, sum-of-exp) pairs into the row's logsumexp. A per-thread
// top-64 in a single round would instead need 256*64*8 = 128 KiB of LDS for the merge, twice
// gfx1201's 64 KiB per-workgroup limit. The logsumexp accumulation is fused into each block's
// round 1 (which re-reads the slice anyway, and by then round 0 has published the slice max), so
// no extra pass is made for it.
//
// This entry point therefore makes TWO device launches and advances the launch counter below by 2.
// It keeps its partials in a module-scope device scratch buffer (~264 KiB of VRAM), so exactly ONE
// call may be in flight per process at a time -- the single-worker-thread assumption model.h's own
// SCOPE comment guarantees for one Model. Two concurrent calls (two threads, two streams -- e.g.
// two tensor-parallel rank threads, docs/tp.md 2.7) would interleave partials; such callers use
// r4dx_topk_lse_f32_ws below, each with its own workspace.
//
// logits: [rows, vocab] fp32 device pointer, rows contiguous (row stride == vocab).
// out_ids: [rows, K] int32. out_vals: [rows, K] fp32. out_lse: [rows] fp32.
// Preconditions (throw std::runtime_error, never a silently wrong answer): 1 <= rows <= 8;
// vocab > K; inv_temperature finite and > 0. A NaN or +inf logit is undefined input, as it is for
// r4dx_topk16_f32 / r4dx_argmax_f32.
enum { R4DX_TOPK_LSE_K = 64 };
void r4dx_topk_lse_f32(int64_t logits, int64_t out_ids, int64_t out_vals, int64_t out_lse,
                        int rows, int64_t vocab, float inv_temperature, int64_t stream);

// The same summary with the per-slice partials in a CALLER-OWNED device workspace instead of the
// module-scope scratch above (docs/tp.md 2.7), so any number of calls may be in flight at once --
// one per workspace. Tensor parallelism needs it: two rank threads summarize their own vocab
// shards concurrently, and under emulation both ranks share ONE device (and so one module scratch).
// `workspace`: device pointer to r4dx_topk_lse_workspace_bytes() bytes, 16-byte aligned (hipMalloc
// alignment is enough); only one call per workspace may be in flight (same-stream calls are
// serialized by stream order). workspace == 0 is the module scratch, i.e. exactly
// r4dx_topk_lse_f32 -- which is what that entry point does, so TP=1 is byte-for-byte unchanged.
// Same preconditions and outputs as r4dx_topk_lse_f32, plus: workspace % 16 == 0.
void r4dx_topk_lse_f32_ws(int64_t logits, int64_t out_ids, int64_t out_vals, int64_t out_lse,
                           int rows, int64_t vocab, float inv_temperature, int64_t stream,
                           int64_t workspace);
// Bytes r4dx_topk_lse_f32_ws needs in its workspace (the module scratch's size, ~262 KiB).
int64_t r4dx_topk_lse_workspace_bytes();

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

// ==== vision tower device pieces (docs/vision.md, Milestone 8 stage 3) =========================
// The five primitives `Qwen3_5VisionModel`'s forward needs that neither this repo nor
// third_party/libr4d already had. Every linear in the tower is a plain bf16 GEMM
// (r4d_gemm_bf16_nt_m64) and its attention is r4d_attn_vit_h72_bf16, so what is left is the norm,
// the two activations, the axial rope and the learned-position-grid gather -- each with its own
// CPU-reference test in tests/kernels/test_vision_kernels.cpp and its own golden comparison in
// tests/vision/test_vision_tower.cpp.

// ---- LayerNorm with weight AND bias (NOT RMSNorm) ---------------------------------------------
// out[r,:] = (x[r,:] - mean(x[r,:])) * rsqrt(var(x[r,:]) + eps) * weight[:] + bias[:], with the
// BIASED variance (divide by `hidden`, not hidden-1) `torch.nn.LayerNorm` uses. This is a real
// mean-subtracted LayerNorm -- the vision tower's norm1/norm2/merger.norm are `nn.LayerNorm`, not
// the text side's `Qwen3_5RMSNorm`, so r4dx_rmsnorm_bf16 (no mean subtraction, no bias, and a
// `1 + weight` convention on top) is wrong for all three in three separate ways.
// Two reduction passes (mean, then sum of squared deviations) rather than the one-pass
// sum/sum-of-squares identity: at block 26 the residual stream's mean is large relative to its
// variance, and `E[x^2] - E[x]^2` cancels catastrophically there in fp32.
// fp32 accumulation and fp32 elementwise math, bf16 in/out, matching torch's own acc_type<bf16>
// = float on CUDA. x: [rows, hidden] bf16. weight, bias: [hidden] bf16. out: [rows, hidden] bf16.
// In-place (out == x) is safe for the same reason r4dx_rmsnorm_bf16's is: both reductions'
// __syncthreads() sit strictly between every read of x and any write to out.
void r4dx_layernorm_bf16(int64_t x, int64_t weight, int64_t bias, int64_t out, int64_t rows,
                          int64_t hidden, float eps, int64_t stream);

// ---- bias add ---------------------------------------------------------------------------------
// out[r,c] = bf16(fp32(x[r,c]) + fp32(bias[c])), the `+ b` half of an `nn.Linear` whose matmul
// half r4d_gemm_bf16_nt_m64 already did. Broadcasts one [cols] row over `rows` rows. In-place
// (out == x) is supported and is what every vision call site uses.
// NOTE on precision: `nn.Linear` adds its bias INSIDE the fp32 GEMM accumulator, so the reference
// rounds to bf16 once; this pair of kernels rounds twice (GEMM output, then here). The extra
// rounding is one bf16 ulp of the pre-bias sum, ~2^-9 relative -- far below the 1e-2 relative-L2
// band docs/vision.md validates against, and measured as such (docs/vision.md "Measured").
// x: [rows, cols] bf16. bias: [cols] bf16. out: [rows, cols] bf16.
void r4dx_bias_add_bf16(int64_t x, int64_t bias, int64_t out, int64_t rows, int64_t cols,
                         int64_t stream);

// ---- GELU, both variants ----------------------------------------------------------------------
// The vision tower uses BOTH, at different sites, and they are not interchangeable:
//   * the encoder MLP's activation is `gelu_pytorch_tanh` (vision_config.hidden_act), i.e. the
//     tanh approximation 0.5*x*(1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)));
//   * the merger's activation is a bare `nn.GELU()`, i.e. approximate='none', the EXACT
//     0.5*x*(1 + erf(x/sqrt(2))).
// They differ by up to ~1e-3 absolute around |x| ~ 2, which is well inside what a "looks
// plausible" check would miss and outside the merger's own tolerance, so they are two entry
// points rather than one with a flag a call site could get wrong silently.
// fp32 math, bf16 in/out, elementwise over `n` elements. In-place (out == x) is supported.
void r4dx_gelu_tanh_bf16(int64_t x, int64_t out, int64_t n, int64_t stream);
void r4dx_gelu_erf_bf16(int64_t x, int64_t out, int64_t n, int64_t stream);

// ---- vision axial rope + qkv split -------------------------------------------------------------
// `Qwen3_5VisionAttention.forward`'s first two steps in one launch: split the fused qkv projection
// into three contiguous [tokens, heads, head_dim] tensors (the layout r4d_attn_vit_h72_bf16 takes)
// and rotate q and k by the axial rope.
//
// The fused row layout is the reference's own `reshape(seq, 3, heads, head_dim)`: within row t,
// q is columns [0, heads*head_dim), k is [heads*head_dim, 2*heads*head_dim), v is the last third.
//
// Rope math, per (token, head), exactly `apply_rotary_pos_emb_vision`: computed in FP32 (the
// reference explicitly upcasts q/k/cos/sin to float32 before rotating, and rounds back afterwards),
// over the FULL head_dim -- there is no partial_rotary_factor here, unlike the text side's 0.25 --
// with the rotate-half pairing `cat(-x2, x1)`:
//   half = head_dim/2
//   out[i]        = x[i]        * cos[i]        - x[i + half] * sin[i]          (i < half)
//   out[i + half] = x[i + half] * cos[i + half] + x[i]        * sin[i + half]
// cos/sin are per TOKEN, shared across heads, and already carry the axial
// cat([f_h, f_w, f_h, f_w]) recomposition (src/vision/vision_index.cpp's BuildVisionRopeCosSin).
// v is copied through unrotated, which is the whole of what the reference does to it.
//
// qkv: [tokens, 3*heads*head_dim] bf16 -- the GEMM output with its bias ALREADY added.
// cos, sin: [tokens, head_dim] fp32. q_out, k_out, v_out: [tokens, heads, head_dim] bf16.
// Precondition (throws): head_dim even.
void r4dx_vision_qkv_rope_bf16(int64_t qkv, int64_t cos, int64_t sin, int64_t q_out, int64_t k_out,
                                int64_t v_out, int tokens, int heads, int head_dim, int64_t stream);

// ---- learned position-embedding gather (4-tap bilinear) ----------------------------------------
// `Qwen3_5VisionModel.forward`'s
//   pos_embeds = (pos_embed(interp_indices) * interp_weights[:, :, None]).sum(1)
//   hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)
// in one launch: per patch, gather `taps` rows of the learned [num_grid_per_side^2, hidden] table
// and weighted-sum them. The table is bf16 and the weights fp32, so the reference's product
// promotes to fp32 and the sum is fp32 -- reproduced here, including the fact that the fp32 result
// is rounded to bf16 BEFORE the residual add (`.to(hidden_states.dtype)` happens first), not after.
//
// Both outputs are optional and independent, so one launch serves the model path and the test:
//   out_f32   != 0 -> [num_patches, hidden] fp32, the raw weighted sum (the golden's `pos_embeds`).
//   out_bf16  != 0 -> [num_patches, hidden] bf16 = bf16(fp32(x[p,:]) + fp32(bf16(sum))); requires
//                     `x` != 0. May alias `x`.
// table: [table_rows, hidden] bf16. indices: [num_patches, taps] int32, flat row offsets into the
// table. weights: [num_patches, taps] fp32. x: [num_patches, hidden] bf16.
// Bounds-checked the same way r4dx_embedding_gather_bf16 is: an index outside [0, table_rows)
// clamps to row 0 rather than reading out-of-bounds device memory.
void r4dx_vision_pos_embed_bf16(int64_t table, int64_t indices, int64_t weights, int64_t x,
                                 int64_t out_f32, int64_t out_bf16, int64_t num_patches,
                                 int64_t hidden, int taps, int64_t table_rows, int64_t stream);

// ---- kernel launch counter (docs/r9700.md P2/task item 4, 2026-09-20) -------------------------
// A process-global counter (a relaxed std::atomic since docs/tp.md 2.7: two tensor-parallel rank
// threads launch concurrently in one process; with them it counts both ranks' launches together)
// incremented once per r4dx-owned kernel launch above (every r4dx_* entry point in this header,
// AFTER its `rows/M/T <= 0` early-return check, so a no-op call does not count). Counts ONLY
// r4dx-owned launches -- NOT the r4d_gemm_*/r4d_gdn_*/r4d_attn_* launches in third_party/libr4d,
// which this project does not instrument (out of scope: a third_party submodule). Used by
// Model::DecodeStepProfiled (model.cpp) to report "r4dx-owned kernel launches per token" before/
// after the R3/P2 fusion pass -- see docs/mtp.md and docs/status.md for the measured before/after.
void r4dx_kernel_launch_counter_reset();
int64_t r4dx_kernel_launch_counter_get();

}  // extern "C"
