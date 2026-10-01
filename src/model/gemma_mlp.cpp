#include "gemma_mlp.h"

#include "linear.h"
#include "profile_span.h"
#include "r4dx/kernels/gemma_kernels.h"
#include "r4dx/kernels/kernels.h"
#include "r4dx/kernels/rotate_residual.h"
#include "rotation_meta.h"  // kHadDownBlock

namespace r4dx::model {

void GemmaMlp::Forward(core::Stream& stream, core::Arena& arena, const uint16_t* x_normed, uint16_t* down_out,
                       int64_t T, SpanAccumulator* prof) {
  const int64_t hidden = cfg_.hidden_size;
  const int64_t inter = cfg_.intermediate_size;
  const int64_t s = reinterpret_cast<int64_t>(stream.get());
  const hipStream_t s_raw = stream.get();
  const auto P = [](const void* p) { return reinterpret_cast<int64_t>(p); };

  uint16_t* gate_up = arena.Alloc<uint16_t>(static_cast<size_t>(T * 2 * inter), 16);
  ProfiledCall(prof, s_raw, "gemm:mlp.gate_up", [&] { ApplyLinear(stream, arena, w_.gate_up, x_normed, gate_up, T); });

  // A trellis mlp.down takes the product straight into its input transform (the fused producer, byte-identical
  // to the v1 pair below), exactly as Qwen's Mlp does with silu.
  const bool trellis_fused = down_had_signs_ == nullptr && TrellisFusionEnabled() &&
                             w_.down.layout == Layout::kTrellis && w_.down.trellis_parts == 1 && w_.down.K == inter;
  uint16_t* h = nullptr;
  PreQuantizedActivation pre;
  if (trellis_fused) {
    uint16_t* a = arena.Alloc<uint16_t>(static_cast<size_t>(T * inter), 16);
    ProfiledCall(prof, s_raw, "mlp.gelu_mul_trellis", [&] {
      r4dx_gelu_tanh_mul_trellis_bf16(P(gate_up), T, inter, /*in_row_stride=*/2 * inter,
                                      P(w_.down.trellis_suh.data()), P(a), w_.down.trellis_prescale_log2, s);
    });
    h = a;
    pre = PreQuantizedActivation{r4dx_epilogue_none, a, w_.down.trellis_suh.data(), 0};
  } else {
    h = arena.Alloc<uint16_t>(static_cast<size_t>(T * inter), 16);
    ProfiledCall(prof, s_raw, "mlp.gelu_mul", [&] {
      r4dx_gelu_tanh_mul_wide_bf16(P(gate_up), P(h), T, inter, /*in_row_stride=*/2 * inter, s);
    });
    if (down_had_signs_ != nullptr) {
      // Rotated container: mlp.down was folded W Hb (512-wide blocks), so h is rotated in place first.
      ProfiledCall(prof, s_raw, "mlp.down_hadamard", [&] {
        r4dx_hadamard_inplace_bf16(P(h), T, inter, P(down_had_signs_), static_cast<int>(kHadDownBlock), s);
      });
    }
  }
  ProfiledCall(prof, s_raw, "gemm:mlp.down", [&] {
    ApplyLinear(stream, arena, w_.down, h, down_out, T, trellis_fused ? &pre : nullptr);
  });
}

}  // namespace r4dx::model
