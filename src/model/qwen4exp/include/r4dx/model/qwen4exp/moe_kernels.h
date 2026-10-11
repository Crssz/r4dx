// qwen4_exp router + MoE data-movement kernels (HF Qwen4ExpTextTopKRouter = Qwen3NextTopKRouter, qwen3_next
// modeling:826-844; Qwen4ExpTextSparseMoeBlock = Qwen3NextSparseMoeBlock:847-866 with Qwen3NextExperts:787-823).
// docs/qwen4exp-plan.md 4.7, 4.11 item 5. Plain C ABI like third_party/libr4d/r4d.h (see hc_kernels.h): device
// pointers are `const void*` / `void*`, the return value a hipError_t (0 on success).
//
// What HF does (and these kernels reproduce):
//   logits = F.linear(x, router_w)                              bf16 [rows, E]       (a GEMM, not here)
//   probs  = softmax(logits, dtype=float32)                     fp32
//   top-k of probs (torch.topk, sorted descending); norm_topk_prob: v /= v.sum(-1)   fp32
//   scores = v.to(bf16)                                         <- weights are bf16-rounded before use
//   per expert: h = Rb(silu(gate)) * up  (bf16), y = down(h) * score, index_add_ into the bf16 output
//   out = experts_out + Rb(sigmoid(shared_gate(x))) * shared_expert(x)
#pragma once

#include <cstdint>

extern "C" {

// Router softmax + top-k + renormalisation + per-expert histogram. One wave per row.
//   logits  bf16 [rows, experts] (the router GEMM output; must be finite)
//   ids     int32 [rows, k]      selected experts, ordered by descending score
//   weights fp32 [rows, k]       routing weights: probs of the selected experts (fp32 softmax of the bf16 logits),
//                                divided by their sum when norm_topk, then rounded to bf16 (the value is stored
//                                widened to fp32) -- HF `router_top_value.to(router_logits.dtype)`
//   counts  int32 [experts]      atomically incremented once per selected (row, expert); the caller zeroes it first
// TIE RULE (documented because torch.topk's own order among equal values is implementation defined, and bf16 logits
// do tie, most often at the k-th / (k+1)-th cut): the larger logit wins; among EQUAL logits the LOWER expert id wins,
// both for membership of the top-k and for the order of ids. Selection compares the bf16 logit values themselves
// (softmax is strictly monotone, and equal logits give equal probabilities), so no fp32 exp rounding can reorder or
// tie anything. experts <= 1024, k <= 16, k <= experts.
int q4x_moe_route_bf16(const void* logits, int32_t* ids, float* weights, int32_t* counts, int64_t rows, int experts,
                       int k, int norm_topk, void* stream);

// out[p, :] = x[row_idx[p], :] for p in [0, n): the rows of the routed (row, expert) pairs, sorted by expert, as one
// contiguous [n, hidden] block (expert e owns the slice [offset[e], offset[e+1]) of it). bf16.
int q4x_moe_gather_rows_bf16(const void* x, const int32_t* row_idx, void* out, int64_t n, int hidden, void* stream);

// out[r, i] = bf16( Rb(silu(gate[r, i])) * up[r, i] ), i in [0, inter): HF `act_fn(gate) * up` where act_fn(gate) is
// a bf16 tensor. gate / up are bf16 matrices with row strides `gate_stride` / `up_stride` (elements), so the fused
// expert gate_up output [n, 2 * inter] (gate = base, up = base + inter, both strides 2 * inter) and the shared
// expert's two separate outputs both fit. out: bf16 [rows, inter] contiguous. (src/kernels' r4dx_silu_mul_bf16 does
// not round silu(gate) to bf16 before the product, so it is not bit-compatible with HF; this one is.)
int q4x_moe_silu_mul_bf16(const void* gate, const void* up, void* out, int64_t rows, int inter, int64_t gate_stride,
                          int64_t up_stride, void* stream);

// Weighted scatter-add of the expert outputs plus the gated shared expert, rounded to bf16 ONCE, as a deterministic
// gather per output row (no atomics, no fp32 accumulator buffer):
//   out[r, d] = bf16( sum_{j<k} y_pairs[row_pairs[r*k + j], d] * weights[row_w_idx[r*k + j]]            fp32, j ascending
//                     + Rb( Rb(sigmoid(shared_gate_logit[r])) * shared_y[r, d] ) )
// row_pairs[r*k + j] is the position in the expert-sorted pair list (the row of y_pairs) of row r's j-th selected
// expert in ASCENDING EXPERT ID order -- the order torch's expert loop adds in -- and row_w_idx[r*k + j] the index
// (r*k + slot) of its routing weight in `weights`. y_pairs: bf16 [P, hidden] (the down outputs, expert-sorted).
// shared_y: bf16 [rows, hidden], shared_gate_logit: bf16 [rows]; both null = no shared expert. out: bf16 [rows, hidden].
int q4x_moe_combine_bf16(const void* y_pairs, const int32_t* row_pairs, const int32_t* row_w_idx,
                         const float* weights, const void* shared_y, const void* shared_gate_logit, void* out,
                         int64_t rows, int k, int hidden, void* stream);

}  // extern "C"
