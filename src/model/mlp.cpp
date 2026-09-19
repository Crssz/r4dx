#include "mlp.h"

#include "linear.h"
#include "profile_span.h"
#include "r4dx/kernels/kernels.h"

namespace r4dx::model {

void Mlp::Forward(core::Stream& stream, core::Arena& arena, const uint16_t* x, uint16_t* x_out,
                   int64_t T, const uint16_t* x_normed_in, const uint16_t* next_norm_weight,
                   uint16_t* x_normed_out, SpanAccumulator* prof) {
  const int64_t hidden = cfg_.hidden_size;
  const int64_t intermediate = cfg_.intermediate_size;
  const float eps = static_cast<float>(cfg_.rms_norm_eps);
  const int64_t s = reinterpret_cast<int64_t>(stream.get());
  const hipStream_t s_raw = stream.get();

  // R3 (docs/r9700.md): skip this block's own rmsnorm launch when the producer already fused it
  // into its residual-add epilogue (r4dx_residual_rmsnorm_bf16) and handed us the result.
  const uint16_t* x_normed = x_normed_in;
  uint16_t* x_normed_scratch = nullptr;
  if (x_normed == nullptr) {
    x_normed_scratch = arena.Alloc<uint16_t>(static_cast<size_t>(T * hidden));
    ProfiledCall(prof, s_raw, "mlp.rmsnorm", [&] {
      r4dx_rmsnorm_bf16(reinterpret_cast<int64_t>(x),
                         reinterpret_cast<int64_t>(post_attention_layernorm_.data()),
                         reinterpret_cast<int64_t>(x_normed_scratch), T, hidden, eps, s);
    });
    x_normed = x_normed_scratch;
  }

  uint16_t* gate_up = arena.Alloc<uint16_t>(static_cast<size_t>(T * 2 * intermediate));
  ProfiledCall(prof, s_raw, "gemm:mlp.gate_up", [&] {
    ApplyLinear(stream, arena, w_.gate_up, x_normed, gate_up, T);
  });

  uint16_t* h = arena.Alloc<uint16_t>(static_cast<size_t>(T * intermediate));
  ProfiledCall(prof, s_raw, "mlp.silu_mul", [&] {
    r4dx_silu_mul_bf16(reinterpret_cast<int64_t>(gate_up), reinterpret_cast<int64_t>(h), T,
                        intermediate, /*in_row_stride=*/2 * intermediate, s);
  });

  uint16_t* down_out = arena.Alloc<uint16_t>(static_cast<size_t>(T * hidden));
  ProfiledCall(prof, s_raw, "gemm:mlp.down", [&] {
    ApplyLinear(stream, arena, w_.down, h, down_out, T);
  });

  ProfiledCall(prof, s_raw, "mlp.residual", [&] {
    if (next_norm_weight != nullptr) {
      r4dx_residual_rmsnorm_bf16(reinterpret_cast<int64_t>(x), reinterpret_cast<int64_t>(down_out),
                                  reinterpret_cast<int64_t>(next_norm_weight),
                                  reinterpret_cast<int64_t>(x_out),
                                  reinterpret_cast<int64_t>(x_normed_out), T, hidden, eps, s);
    } else {
      r4dx_residual_add_bf16(reinterpret_cast<int64_t>(x), reinterpret_cast<int64_t>(down_out),
                              reinterpret_cast<int64_t>(x_out), T * hidden, s);
    }
  });
}

}  // namespace r4dx::model
