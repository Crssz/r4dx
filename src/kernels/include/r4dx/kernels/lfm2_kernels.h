// LFM2.5-24B-A2B device pieces (docs/lfm2-plan.md 3.3 ShortConv, 3.5 MoE; task L1-6). Same plain C ABI as
// gemma_kernels.h: device pointers and the stream as int64_t, shapes/strides in ELEMENTS, bf16 in/out with
// fp32 math. Implemented in src/kernels/src/lfm2_kernels.hip (its own hipcc object, exactly
// r4dx_kernels.hip's flags, -ffp-contract=off among them, so every product below is a separately rounded
// multiply in the order written). Every entry point counts into r4dx_kernel_launch_counter (after its
// early return for empty work). CPU references: tests/kernels/lfm2_ref.hpp.
//
// head_dim 64 needs NO new kernel: r4dx_rope_neox_bf16 (kernels.h; head_dim any even value, one thread per
// frequency bin) and r4dx_rmsnorm_plain_bf16 (hidden any, 256 threads stride the row) already handle 64;
// the GPU test test_lfm2_rope_norm64 pins both at the LFM geometry (rope theta 1e7, per-head qk norm over
// rows T * heads with hidden 64).
#pragma once

#include <cstdint>

extern "C" {

// ---- fused ShortConv (decode T = 1 and chunked prefill T >= 1) --------------------------------------
// Lfm2ShortConv without the two projections (in_proj / out_proj stay ordinary GEMMs):
//   B | C | x = bcx[t, 0..H) | [H..2H) | [2H..3H)
//   h_t   = bf16(B_t * x_t)
//   acc_t = (w0 * h_{t-2} + w1 * h_{t-1}) + w2 * h_t           (fp32 over bf16 operands, this order)
//   out_t = bf16(C_t * bf16(acc_t))
// with h_{-2}, h_{-1} = state[0], state[1] (zeros for a fresh sequence: HF's zero causal padding). The
// state is UPDATED IN PLACE in the same call (stream order after every read of the old one): afterwards
// state[0] = h at the second-to-last row of the chunk, state[1] = h at the last row; for T == 1 that is
// the shift state[0] <- old state[1], state[1] <- h_0. Feeding a sequence as one chunk of N rows, as N
// chunks of 1 row, or in any split gives the same bytes (the math is a function of h only).
//   bcx    : [T, 3H] bf16 (row stride bcx_row_stride >= 3H elements), read-only
//   state  : [2, H] bf16, in/out (state[0] = tap oldest .. state[1] = h at t-1)
//   weight : [H, 3] bf16 (taps oldest..newest: weight[c, 0] multiplies h_{t-2}, [c, 2] multiplies h_t)
//   out    : [T, H] bf16, row stride H
// One launch when T <= 16, otherwise the conv launch plus a 2-column state micro-kernel. Preconditions
// (throw): H >= 1, bcx_row_stride >= 3H; T <= 0 returns.
void r4dx_lfm2_shortconv_bf16(int64_t bcx, int64_t state, int64_t weight, int64_t out, int T, int H,
                               int64_t bcx_row_stride, int64_t stream);

// ---- sigmoid router (fp32) ----------------------------------------------------------------------------
// One workgroup per token row. logits[e] = sum_k float(x[k]) * float(gate_w[e, k]) in fp32;
//   s[e] = sigmoid(logits[e]) (fp32, 1 / (1 + expf(-l)));  score[e] = s[e] + expert_bias[e] (fp32)
//   sel = top-k of score, descending, ties -> LOWER expert index (so the slot order is deterministic; HF's
//   topk(sorted=False) order is unspecified and irrelevant to the fp32 combine)
//   w[slot] = s[sel[slot]] / (sum over the k selected s (slot order, fp32) + 1e-6) * routed_scaling_factor
// (HF Lfm2MoeSparseMoeBlock: the bias moves the SELECTION only, the weights are the unbiased sigmoids,
// normalized by their sum + 1e-6 and then scaled.) Replicated ranks see the same bits and pick the same
// experts. x: [T, hidden] bf16 (row stride x_row_stride >= hidden). gate_w: [n_experts, hidden] bf16.
// expert_bias: [n_experts] fp32. ids: [T, top_k] int32. w: [T, top_k] fp32.
// Preconditions (throw): 1 <= n_experts <= 64, 1 <= top_k <= min(8, n_experts), 1 <= hidden <= 8192.
void r4dx_lfm2_router_f32(int64_t x, int64_t gate_w, int64_t expert_bias, int64_t ids, int64_t w,
                           int T, int hidden, int n_experts, int top_k, float routed_scaling_factor,
                           int64_t x_row_stride, int64_t stream);

// ---- expert counting sort (one workgroup; deterministic) ------------------------------------------------
// Sorts the P = T * top_k (token, slot) pairs by expert. A pair is the flat index p = t * top_k + slot into
// the router's [T, top_k] arrays. The order inside an expert is ascending p (= token order), independent of
// scheduling: no atomics.
//   expert_counts : [n_experts] int32, rows per expert
//   expert_offsets: [n_experts + 1] int32, exclusive prefix sum of the counts (offsets[n_experts] = number
//                   of valid pairs)
//   sorted_pairs  : [P] int32; sorted_pairs[offsets[e] .. offsets[e+1]) are the pairs routed to e
//   pos_of_pair   : [P] int32, the inverse map: sorted_pairs[pos_of_pair[p]] == p
// A pair whose id is outside [0, n_experts) is skipped (pos_of_pair = -1, not counted). The caller reads
// expert_counts back once per layer to drive the per-expert GEMMs (rows [offsets[e], offsets[e+1]) of the
// gathered buffer). Preconditions (throw): 1 <= n_experts <= 256, 1 <= P <= 16384.
void r4dx_lfm2_moe_sort(int64_t ids, int T, int top_k, int n_experts, int64_t expert_counts,
                         int64_t expert_offsets, int64_t sorted_pairs, int64_t pos_of_pair,
                         int64_t stream);

// out[i, :] = x[sorted_pairs[i] / top_k, :] for i in [0, n_pairs): the expert-sorted input rows. x:
// [T, hidden] bf16 (row stride x_row_stride), out: [n_pairs, hidden] bf16 contiguous. n_pairs = T * top_k;
// entries of sorted_pairs at or beyond expert_offsets[n_experts] (skipped pairs) are not read when
// valid_pairs (a DEVICE int32 pointer, e.g. expert_offsets + n_experts) is non-zero, and the matching out
// rows are left untouched; pass 0 to gather all n_pairs.
void r4dx_lfm2_moe_gather_bf16(int64_t x, int64_t sorted_pairs, int64_t out, int n_pairs, int top_k,
                                int hidden, int64_t x_row_stride, int64_t valid_pairs, int64_t stream);

// ---- MoE combine (fp32, fixed order) --------------------------------------------------------------------
// out[t, :] = bf16( sum_{slot = 0..top_k-1, in this order} w[t, slot] * float(y_row(t, slot)[:]) ), fp32
// accumulation from 0.0f, one rounding. No atomics; the result does not depend on scheduling.
//   y_row(t, slot) = y[t * top_k + slot]            when pos_of_pair == 0   ([T, top_k, hidden] slot layout,
//                                                    the M = 1 decode path writes 4 rows)
//                  = y[pos_of_pair[t*top_k+slot]]    otherwise (expert-sorted layout of r4dx_lfm2_moe_sort;
//                                                    a negative position contributes nothing)
// y: bf16 rows of `hidden` contiguous elements. w: [T, top_k] fp32. out: [T, hidden] bf16 contiguous.
void r4dx_lfm2_moe_combine_bf16(int64_t y, int64_t pos_of_pair, int64_t w, int64_t out, int T,
                                 int top_k, int hidden, int64_t stream);

// ---- grouped MoE decode GEMM (bf16; docs/lfm2-plan.md 3.5 / L4-2) ---------------------------------------
// Device-indexed M = 1 GEMM across n_pairs = T * top_k routed (token, slot) pairs in one launch
// (grid.z = p in 0..n_pairs-1). For pair p:
//   expert e = ids[p] (int32 on device; a pair with e outside [0, n_experts) returns without writing C);
//   weight W = (const bf16*)w_ptrs[e] ([N, K] row-major bf16, from the layer's device uint64 pointer table);
//   input  A = a_bf16 + (p / a_row_group) * K (a_row_group = top_k for gate_up, which shares token row
//              t = p / top_k across its top_k slots; a_row_group = 1 for down, where each pair p has its
//              own [1, K] row);
//   output C = c_bf16 + p * N ([1, N] bf16).
// Each pair p executes the exact M = 1, MT = 1 WMMA loop and fixed-order LDS split-K reduction of
// r4d_gemm_bf16_nt_m64, so C[p, :] is bit-for-bit identical to calling r4d_gemm_bf16_nt_m64(A_p, W_e, C_p,
// 1, K, N, WV, SK, 1, stream) at the same (WV, SK). Counts 1 launch in r4dx_kernel_launch_counter.
// Preconditions (throw): 1 <= n_pairs <= 16384, 1 <= a_row_group <= n_pairs, n_pairs % a_row_group == 0,
// 1 <= n_experts <= 256, N >= 16, N % 16 == 0, WV >= 1, SK >= 1, K >= 16, K % (SK * 16) == 0,
// WV * SK * 32 <= 1024, WV * SK * 256 * sizeof(float) <= 64 * 1024, non-null pointers.
void r4dx_lfm2_gemm_bf16_moe_decode(int64_t a_bf16, int64_t w_ptrs, int64_t ids, int64_t c_bf16,
                                    int n_pairs, int a_row_group, int n_experts, int K, int N,
                                    int WV, int SK, int64_t stream);

// ---- grouped MoE decode input transform + GEMM (trellis; docs/lfm2-plan.md 3.5 / L4-3) ------------------
// Device-indexed input transform across n_pairs = T * top_k routed (token, slot) pairs (grid.y = p):
//   expert e = ids[p] (int32 on device; a pair with e outside [0, n_experts) returns without writing);
//   suh_e    = (const float*)suh_ptrs[e] (float [parts * K]);
//   x_p      = x_bf16 + (p / a_row_group) * K;
//   a0_p     = a0_f16 + p * K; a1_p = a1_f16 + p * K (when parts == 2).
// Bit-for-bit identical to calling r4dx_trellis_input_bf16(x_p, 1, K, parts, {suh_e, suh_e + K},
// {a0_p, a1_p}, prescale_log2, stream) per pair. Counts 1 launch in r4dx_kernel_launch_counter.
void r4dx_lfm2_trellis_moe_input_bf16(int64_t x_bf16, int64_t suh_ptrs, int64_t ids, int64_t a0_f16,
                                      int64_t a1_f16, int n_pairs, int a_row_group, int n_experts,
                                      int64_t K, int parts, int prescale_log2, int64_t stream);

// Device-indexed M = 1 trellis GEMM across n_pairs = T * top_k routed (token, slot) pairs (grid.z = p):
//   expert e = ids[p] (int32 on device; a pair with e outside [0, n_experts) returns without writing C);
//   W_e      = (const uint32_t*)w_ptrs[e], svh_e = (const float*)svh_ptrs[e];
//   A0_p     = a0_f16 + p * K, A1_p = a1_f16 ? a1_f16 + p * K : 0;
//   C_p      = c_bf16 + p * N;
//   ws_p     = ws_f32 ? ws_f32 + p * SKG * N : 0;
//   tickets_p = tickets_u32 ? tickets_u32 + p * (N / 128) : 0 (disjoint per pair, self-resetting).
// Bit-for-bit identical to calling r4d_gemm_trellis_nt_m64(A0_p, A1_p, n_split, W_e, svh_e, C_p, ws_p,
// tickets_p, 1, K, N, bits, WV, SK, 1, NP, SKG, U, NT, out_scale, stream) per pair.
// Instantiated for bits in {4, 5}, NP = 1, U in {1, 2, 4}, NT in {0, 1}. Counts 1 launch.
void r4dx_lfm2_gemm_trellis_moe_decode(int64_t a0_f16, int64_t a1_f16, int n_split, int64_t w_ptrs,
                                       int64_t svh_ptrs, int64_t ids, int64_t c_bf16, int64_t ws_f32,
                                       int64_t tickets_u32, int n_pairs, int n_experts, int K, int N,
                                       int bits, int WV, int SK, int NP, int SKG, int U, int NT,
                                       float out_scale, int64_t stream);

// ---- grouped MoE prefill GEMM (bf16; docs/lfm2-plan.md 3.5 / L4-4) --------------------------------------
// Device-indexed grouped GEMM over the expert-sorted rows A[n_pairs, K] -> C[n_pairs, N], driven by
// device `expert_offsets[n_experts + 1]` (from r4dx_lfm2_moe_sort) with zero host readback.
// Grid.y enumerates 16-row tiles across all non-empty experts (at most min(n_pairs, (n_pairs+15)/16 + n_experts)
// tiles); each active tile (expert e, row_start = offsets[e] + 16*mt, m_rows = min(16, offsets[e+1] - row_start))
// runs the exact MT = 1 WMMA + LDS split-K reduction of r4d_gemm_bf16_nt_m64 on W = w_ptrs[e], producing
// output bytes identical to calling r4d_gemm_bf16_nt_m64 per non-empty expert at the same SK.
void r4dx_lfm2_gemm_bf16_moe_prefill(int64_t a_bf16, int64_t w_ptrs, int64_t expert_offsets,
                                     int64_t c_bf16, int n_pairs, int n_experts, int K, int N,
                                     int WV, int SK, int64_t stream);

// ---- L4-5 fusions (docs/lfm2-plan.md L4-5) ---------------------------------------------------------------
// Fused in-place residual add + plain RMSNorm:
//   h[r, i] = bf16(h[r, i] + sub[r, i])
//   x_out[r, i] = bf16(float(h[r, i]) * rsqrt(mean(float(h[r, :])**2) + eps) * float(weight[i]))
// Bit-for-bit identical to calling r4dx_residual_add_bf16(h, sub, h, rows * hidden, stream) followed by
// r4dx_rmsnorm_plain_bf16(h, weight, x_out, rows, hidden, eps, 0, stream). Counts 1 launch.
void r4dx_lfm2_residual_rmsnorm_plain_bf16(int64_t h, int64_t sub, int64_t weight, int64_t x_out,
                                           int64_t rows, int64_t hidden, float eps, int64_t stream);

// Fused attention pre-core (head_dim == 64): per-head plain RMSNorm on q[T, heads_q, 64] and
// k[T, heads_k, 64], NeoX rotate_half RoPE on q and k at positions pos_ids[T], and paged bf16 KV cache
// write of (roped k, raw v) at slots pos_ids[T] -- all in one launch of (T, heads_q + heads_k) workgroups.
// Bit-for-bit identical to the 4-launch chain (q_norm, k_norm, rope_neox, kv_write_paged_bf16_hnd).
// Counts 1 launch. Preconditions (throw): head_dim == 64, heads_q >= 1, heads_k >= 1, block_size >= 1.
void r4dx_lfm2_attn_precore_bf16(int64_t q, int64_t k, int64_t v, int64_t q_norm, int64_t k_norm,
                                 int64_t pos_ids, int64_t kv_cache, int T, int heads_q, int heads_k,
                                 int head_dim, float rms_eps, float rope_theta, int block_size,
                                 int64_t kv_block_stride, int64_t kv_head_stride, int64_t stream);

}  // extern "C"
