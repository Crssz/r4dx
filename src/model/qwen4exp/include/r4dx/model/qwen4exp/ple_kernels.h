// qwen4_exp PLE layer kernels (HF Qwen4ExpTextPLELayer, modular_qwen4_exp.py:711-783; docs/qwen4exp-plan.md 2.3, 4.8).
// Plain C ABI (see hc_kernels.h). The two projections and the three grouped norms around these kernels are issued by
// PleLayer (ple_layer.h); the norms are q4x_grouped_rmsnorm_bf16 of hc_kernels.h.
//
// HF forward, with C = hc * hidden = 10240 channels, per token:
//   key_normed   = norm_key(key_proj(emb))              per stream g: [hidden]
//   value        = value_proj(emb)                      [hidden]
//   query_normed = norm_query(h)                        h = the residual BEFORE this layer adds the PLE output
//   gate  = sum_d(key_normed * query_normed) / sqrt(hidden)           per stream
//   gate  = sqrt(max(|gate|, 1e-6)) * sign(gate)
//   gated_value = sigmoid(gate) * value                  [C], value broadcast over the streams
//   out = gated_value + silu(dilated_depthwise_conv(norm_conv(gated_value)))
//   h  += out                                            (decoder layer, :811-814)
#pragma once

#include <cstdint>

extern "C" {

// Gate and gated value. For each (row, stream g), hidden = group size:
//   kn   = Rb(key_proj_out * rsqrt(mean(key_proj_out^2) + eps) * (1 + norm_key_w))       (group of stream g)
//   qn   = Rb(h * rsqrt(mean(h^2) + eps) * (1 + norm_query_w))
//   gate = Rb(Rb(sum_d Rb(kn * qn)) / sqrt(hidden)); gate = Rb(sqrt(Rb(max(|gate|, 1e-6)))) * sign(gate)
//   gated_value[r, g*hidden + d] = bf16( Rb(sigmoid(gate)) * value[r, d] )
// i.e. every HF bf16 tensor boundary is rounded; the sums accumulate in fp32 (sum over d ascending within a thread,
// then a tree across the workgroup).
// key_proj_out, h, gated_value: bf16 [rows, hc * hidden]; value: bf16 [rows, hidden]; norm_*_w: bf16 [hc * hidden]
// raw weights. hidden <= 4096. gated_value must not alias an input.
int q4x_ple_gate_value_bf16(const void* key_proj_out, const void* h, const void* value, const void* norm_key_w,
                            const void* norm_query_w, void* gated_value, int64_t rows, int hc, int hidden, float eps,
                            void* stream);

// Dilated depthwise causal conv (kernel 4, dilation 3) over the norm_conv'd gated value with a persistent state, silu,
// the residual of the layer and the residual add, one thread per channel stepping through the rows sequentially:
//   window of channel c at row t = [state[c][0..8], gvn[0..t]] shifted so its newest element is gvn[t];
//   conv = Rb(w[c][0] * x[t-9] + w[c][1] * x[t-6] + w[c][2] * x[t-3] + w[c][3] * x[t])        (fp32 sum, tap 0 first)
//   h[t, c] = bf16( h[t, c] + Rb( gated_value[t, c] + Rb(silu(conv)) ) )
// and afterwards state[c] = the last 9 values of [state, gvn] (so chunks of any size, including 1, compose exactly: the
// result of processing 1 + 6 + 1 rows equals processing the 8 at once, bit for bit).
// gvn, gated_value: bf16 [rows, channels]; conv_w: bf16 [channels, kernel] (the HF [C, 1, 4] weight); state: bf16
// [channels, (kernel - 1) * dilation], oldest step first, read and written in place; h: bf16 [rows, channels] in/out.
// Only kernel = 4, dilation = 3 are instantiated.
int q4x_ple_conv_bf16(const void* gvn, const void* gated_value, const void* conv_w, void* state, void* h,
                      int64_t rows, int channels, int kernel, int dilation, void* stream);

}  // extern "C"
