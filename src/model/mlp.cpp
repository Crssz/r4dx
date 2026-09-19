#include "mlp.h"

#include "linear.h"
#include "r4dx/kernels/kernels.h"

namespace r4dx::model {

void Mlp::Forward(core::Stream& stream, core::Arena& arena, const uint16_t* x, uint16_t* x_out,
                   int64_t T) {
  const int64_t hidden = cfg_.hidden_size;
  const int64_t intermediate = cfg_.intermediate_size;
  const float eps = static_cast<float>(cfg_.rms_norm_eps);
  const int64_t s = reinterpret_cast<int64_t>(stream.get());

  uint16_t* x_normed = arena.Alloc<uint16_t>(static_cast<size_t>(T * hidden));
  r4dx_rmsnorm_bf16(reinterpret_cast<int64_t>(x),
                     reinterpret_cast<int64_t>(post_attention_layernorm_.data()),
                     reinterpret_cast<int64_t>(x_normed), T, hidden, eps, s);

  uint16_t* gate_up = arena.Alloc<uint16_t>(static_cast<size_t>(T * 2 * intermediate));
  ApplyLinear(stream, arena, w_.gate_up, x_normed, gate_up, T);

  uint16_t* h = arena.Alloc<uint16_t>(static_cast<size_t>(T * intermediate));
  r4dx_silu_mul_bf16(reinterpret_cast<int64_t>(gate_up), reinterpret_cast<int64_t>(h), T,
                      intermediate, /*in_row_stride=*/2 * intermediate, s);

  uint16_t* down_out = arena.Alloc<uint16_t>(static_cast<size_t>(T * hidden));
  ApplyLinear(stream, arena, w_.down, h, down_out, T);

  r4dx_residual_add_bf16(reinterpret_cast<int64_t>(x), reinterpret_cast<int64_t>(down_out),
                          reinterpret_cast<int64_t>(x_out), T * hidden, s);
}

}  // namespace r4dx::model
