// qwen4_exp GDN gated RMSNorm with a SIGMOID output gate (HF Qwen4ExpTextRMSNormGated(activation="sigmoid"),
// modular_qwen4_exp.py:313-316 over Qwen3NextRMSNormGated.forward, qwen3_next modeling:66-75; checkpoint
// output_gate_type "sigmoid").
//
// Same interface and shape contract as third_party/libr4d's r4d_gdn_gated_rmsnorm_h128_bf16 (one row per (token, head),
// 128 wide, bf16 x / z / o, fp32 weight [128], strided rows), minus the `act` argument, so GdnLayer-style code can call
// it in place of that one. Differences in numerics, on purpose: libr4d's kernel rounds once at the end
// (bf16(x * rs * w * act(z))); HF rounds twice before the gate, and so does this kernel:
//     norm = Rb(x * rsqrt(mean(x^2) + eps))            hidden_states.to(input_dtype)
//     prod = Rb(w * norm)                              self.weight * (bf16 tensor); w is the bf16 weight widened
//     out  = bf16(prod * sigmoid(z))                   * act(gate.to(float32)), cast back
// (fp32 math for the mean, the rsqrt and the gate.) Note the weight is the plain weight, NOT (1 + w): the gated norm is
// the one norm of the model without the centring.
//
// Only the prefill path of the 27B GDN layer calls a standalone gated norm; the decode kernel
// (r4d_gdn_recurrent_update_*) fuses the same arithmetic into its epilogue and takes an `act` code (0 silu, 1 sigmoid)
// of its own, so decode needs no kernel from here.
#pragma once

#include <cstdint>

extern "C" {

// x [rows][xrow], z [rows][zrow], o [rows][orow]: bf16; w: fp32 [128]; width must be 128. Returns 0, or a hipError_t.
int q4x_gdn_sigmoid_gated_rmsnorm_h128_bf16(const void* x, const void* z, const void* w, void* o, int64_t rows,
                                             int64_t xrow, int64_t zrow, int64_t orow, int width, float eps,
                                             void* stream);

}  // extern "C"
