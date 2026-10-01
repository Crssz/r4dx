// Device-only helpers shared by gemma_kernels.hip and trellis_transform.hip (the GeGLU producer and
// its trellis-fused twin must compute the identical bf16 element). Include from a hipcc TU only.
#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

namespace r4dx::kernels::gemma {

// `gelu_pytorch_tanh`: 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))), the expression (and the fp32
// sqrt(2/pi)) r4dx_gelu_tanh_bf16 uses for the vision tower.
__device__ __forceinline__ float GeluTanh(float x) {
  constexpr float kSqrt2OverPi = 0.7978845608028654f;
  const float inner = kSqrt2OverPi * (x + 0.044715f * x * x * x);
  return 0.5f * x * (1.0f + tanhf(inner));
}

// One GeGLU element: bf16( float(bf16(gelu_tanh(g))) * u ) -- the activation rounds to bf16 before the
// multiply, as `act_fn(gate) * up` does in HF (two bf16 tensor ops).
__device__ __forceinline__ __hip_bfloat16 GeGluElement(float g, float u) {
  const float a = __bfloat162float(__float2bfloat16(GeluTanh(g)));
  return __float2bfloat16(a * u);
}

}  // namespace r4dx::kernels::gemma
