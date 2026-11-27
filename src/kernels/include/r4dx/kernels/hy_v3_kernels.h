// r4dx::kernels -- hy_v3 (Hy-MT3-30B-A3B) optimized device kernels:
//   * r4dx_hy_v3_attn_precore: fused q_norm + k_norm + NeoX RoPE + paged KV write (bf16 or fp8 e4m3)
//   * r4dx_hy_v3_router_f32: multi-block vectorized 128-expert sigmoid+bias router (+ optional shared-expert slot)
//   * r4dx_gemm_w4a16_qkv_bf16a_m1: fused Q+K+V w4a16 M=1 GEMM with bf16 A (1 launch instead of 6)
//   * r4dx_gemm_w4a16_bf16a_m1: single-weight M=1 w4a16 GEMM with bf16 A
//   * r4dx_gemm_w4a16_moe_gate_up_silu_decode: fused grouped w4a16 gate_up + silu_mul (zero LDS, zero syncs)
//   * r4dx_gemm_w4a16_moe_decode / _prefill: device-indexed grouped w4a16 GEMM (bf16 A, zero host syncs)
//   * r4dx_hy_v3_moe_combine_residual_rmsnorm_bf16: vectorized fused MoE combine + residual add + optional next RMSNorm
#pragma once

#include <cstdint>

extern "C" {

void r4dx_hy_v3_attn_precore(int64_t q, int64_t k, int64_t v, int64_t q_norm, int64_t k_norm,
                             int64_t pos_ids, int64_t kv_cache, int kv_fp8,
                             int64_t k_descale, int64_t v_descale,
                             int T, int heads_q, int heads_k, float rms_eps, float rope_theta,
                             int block_size, int64_t kv_block_stride, int64_t kv_head_stride,
                             int64_t stream);

// Multi-block router for T==1 (uses `sig_scratch` [n_experts] floats when non-zero) or single-block for T>1.
// If append_shared != 0, writes (top_k + 1) slots per token: slot [top_k] = expert `n_experts`, weight 1.0f.
void r4dx_hy_v3_router_f32(int64_t x, int64_t gate_w, int64_t expert_bias, int64_t ids, int64_t w,
                           int64_t sig_scratch, int T, int hidden, int n_experts, int top_k,
                           float routed_scaling_factor, int append_shared, int64_t stream);

void r4dx_gemm_w4a16_qkv_bf16a_m1(int64_t a_bf16,
                                  int64_t wq_q, int64_t wsz_q, int64_t c_q, int Nq,
                                  int64_t wq_k, int64_t wsz_k, int64_t c_k, int Nk,
                                  int64_t wq_v, int64_t wsz_v, int64_t c_v, int Nv,
                                  int K, int group, int WV, int SK, int NT, int64_t stream);

void r4dx_gemm_w4a16_bf16a_m1(int64_t a_bf16, int64_t wq, int64_t wsz, int64_t c_bf16,
                              int K, int N, int group, int WV, int SK, int NPW, int NT,
                              int64_t stream);

// Fused grouped w4a16 gate_up + silu(gate)*up for decode (M=1 per pair, N_out = Im, gate_up has 2*Im rows).
void r4dx_gemm_w4a16_moe_gate_up_silu_decode(int64_t a_bf16, int64_t wq_ptrs, int64_t wsz_ptrs,
                                             int64_t ids, int64_t act_bf16, int n_pairs,
                                             int a_row_group, int n_experts, int K, int Im,
                                             int group, int WV, int SK, int NT, int64_t stream);

void r4dx_gemm_w4a16_moe_decode(int64_t a_bf16, int64_t wq_ptrs, int64_t wsz_ptrs, int64_t ids,
                                int64_t c_bf16, int n_pairs, int a_row_group, int n_experts,
                                int K, int N, int group, int WV, int SK, int NPW, int NT,
                                int64_t stream);

// Fused grouped w4a16 MoE down + weighted combine + residual add into h[N] for T==1 (S <= 16 slots, G=64).
void r4dx_gemm_w4a16_moe_down_combine_res_decode(int64_t act_bf16, int64_t wq_ptrs, int64_t wsz_ptrs,
                                                 int64_t ids, int64_t wts, int64_t h_bf16,
                                                 int slots, int n_experts, int K, int N,
                                                 int group, int NT, int64_t stream);

void r4dx_gemm_w4a16_moe_prefill(int64_t a_bf16, int64_t wq_ptrs, int64_t wsz_ptrs,
                                 int64_t expert_offsets, int64_t c_bf16, int n_pairs,
                                 int n_experts, int K, int N, int group, int WV, int SK, int NPW,
                                 int NT, int64_t stream);

void r4dx_hy_v3_rmsnorm_bf16(int64_t h, int64_t norm_w, int64_t out,
                             int T, int hidden, float eps, int64_t stream);

void r4dx_hy_v3_residual_rmsnorm_bf16(int64_t h, int64_t sub, int64_t norm_w, int64_t out,
                                      int T, int hidden, float eps, int64_t stream);

void r4dx_hy_v3_moe_combine_residual_rmsnorm_bf16(int64_t y, int64_t pos_of_pair, int64_t w,
                                                  int64_t h, int64_t next_norm_w, int64_t x_next,
                                                  int T, int slots, int hidden, float eps,
                                                  int64_t stream);

}  // extern "C"
