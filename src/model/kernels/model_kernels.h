// r4dx::model's own two tiny elementwise device kernels -- neither belongs in src/kernels (that
// component's owner is a different agent) and neither exists in r4d.h, but both are required by
// src/model/linear.cpp and src/model/mlp.cpp respectively:
//
//   - r4dx_model_cast_bf16_to_f16: w4a16's activation operand is f16 (r4d_gemm_w4a16_nt_m64's own
//     doc: "f16 A"), while every r4dx activation buffer is produced/consumed as bf16 -- this is
//     the device-side dtype cast docs/architecture.md's "Own kernels" table would have listed had
//     w4a16 existed when that table was written.
//   - r4dx_model_widen_bf16_to_f32: every r4d GEMM kernel (bf16/w4a16/w4a8/mxfp4a8, see each
//     .hip file's epilogue) writes its C operand as bf16 -- there is no fp32-output GEMM variant.
//     The task's "final norm + lm_head -> fp32 logits on device" therefore needs one widen pass
//     after the lm_head GEMM; this is it.
//
// Same plain-C-ABI convention as r4d.h / src/kernels/include/r4dx/kernels/kernels.h: device
// pointers and the stream travel as int64_t, shapes/strides in ELEMENTS. Implemented in
// src/model/kernels/model_kernels.hip, built by hipcc the same way src/kernels/CMakeLists.txt
// builds r4dx_kernels (see src/model/CMakeLists.txt).
#pragma once

#include <cstdint>

extern "C" {

// out[i] = f16(bf16_to_float(in[i])), i in [0, n). in: bf16 (uint16). out: f16 (uint16).
void r4dx_model_cast_bf16_to_f16(int64_t in, int64_t out, int64_t n, int64_t stream);

// out[i] = bf16_to_float(in[i]), i in [0, n). in: bf16 (uint16). out: fp32.
void r4dx_model_widen_bf16_to_f32(int64_t in, int64_t out, int64_t n, int64_t stream);

}  // extern "C"
