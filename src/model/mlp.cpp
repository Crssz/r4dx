#include "mlp.h"

#include "linear.h"
#include "profile_span.h"
#include "r4dx/core/tp_comm.hpp"
#include "r4dx/kernels/kernels.h"
#include "rotation_meta.h"  // kHadDownBlock (quant2 Q2b)

namespace r4dx::model {

void Mlp::Forward(core::Stream& stream, core::Arena& arena, const uint16_t* x, uint16_t* x_out,
                   int64_t T, const uint16_t* x_normed_in, const uint16_t* next_norm_weight,
                   uint16_t* x_normed_out, SpanAccumulator* prof, int x_normed_pre_epilogue,
                   const void* x_normed_pre_data, int next_epilogue,
                   void* next_epilogue_out) {
  const int64_t hidden = cfg_.hidden_size;
  const int64_t intermediate = cfg_.intermediate_size;
  const float eps = static_cast<float>(cfg_.rms_norm_eps);
  const int64_t s = reinterpret_cast<int64_t>(stream.get());
  const hipStream_t s_raw = stream.get();

  // R3 (docs/r9700.md): skip this block's own rmsnorm launch when the producer already fused it
  // into its residual-add epilogue (r4dx_residual_rmsnorm_bf16) and handed us the result.
  // R2/P2 (docs/r9700.md, 2026-09-20): the fused activation-cast epilogue (kernels.h's
  // r4dx_epilogue, f16 only now) is implemented and byte-exact-tested in isolation
  // (tests/kernels/test_fused_quant.cpp) and mechanically wired through
  // GdnLayer::Forward/AttentionLayer::Forward/Mlp::Forward/ApplyLinear. EpilogueForLayout
  // (linear.cpp) returns r4dx_epilogue_none for every layout (w4a16's f16 regressed decode
  // wall-clock, see docs/status.md's R2/P2 section), which makes
  // x_normed_pre_epilogue/next_epilogue always 0 here too, so the code below always takes the plain
  // unfused path; the f16 wiring is left in place so a future pass only has to flip
  // EpilogueForLayout.
  const uint16_t* x_normed = x_normed_in;
  uint16_t* x_normed_scratch = nullptr;
  int gate_up_epilogue = x_normed_pre_epilogue;
  const void* gate_up_pre_data = x_normed_pre_data;
  // Cross-boundary reuse only valid if it matches THIS Mlp's own w_.gate_up layout -- see
  // gdn_layer.cpp's identical defensive comment.
  if (x_normed != nullptr && gate_up_epilogue != EpilogueForLayout(w_.gate_up.layout)) {
    gate_up_epilogue = r4dx_epilogue_none;
    gate_up_pre_data = nullptr;
  }
  if (x_normed == nullptr) {
    gate_up_epilogue = EpilogueForLayout(w_.gate_up.layout);
    x_normed_scratch = arena.Alloc<uint16_t>(static_cast<size_t>(T * hidden));
    uint8_t* gate_up_pre_data_local = nullptr;
    if (gate_up_epilogue != r4dx_epilogue_none) {
      // f16: 2 bytes per element. 16-byte alignment: see attention_layer.hpp's identical comment.
      gate_up_pre_data_local =
          arena.Alloc<uint8_t>(static_cast<size_t>(T * hidden * 2), /*align_bytes=*/16);
    }
    ProfiledCall(prof, s_raw, "mlp.rmsnorm", [&] {
      r4dx_rmsnorm_bf16(reinterpret_cast<int64_t>(x),
                         reinterpret_cast<int64_t>(post_attention_layernorm_.data()),
                         reinterpret_cast<int64_t>(x_normed_scratch), T, hidden, eps, s,
                         gate_up_epilogue, reinterpret_cast<int64_t>(gate_up_pre_data_local));
    });
    x_normed = x_normed_scratch;
    gate_up_pre_data = gate_up_pre_data_local;
  }

  uint16_t* gate_up = arena.Alloc<uint16_t>(static_cast<size_t>(T * 2 * intermediate));
  ProfiledCall(prof, s_raw, "gemm:mlp.gate_up", [&] {
    PreQuantizedActivation pre{gate_up_epilogue, gate_up_pre_data};
    ApplyLinear(stream, arena, w_.gate_up, x_normed, gate_up, T,
                gate_up_epilogue != r4dx_epilogue_none ? &pre : nullptr);
  });

  // R2/P2 (docs/r9700.md): silu_mul already touches every element of its output row, so it emits
  // w_.down's own required cast format directly (fused epilogue, kernels.h's r4dx_epilogue),
  // letting ApplyLinear below skip its separate cast launch for this GEMM. w_.down's layout is
  // known locally (this Mlp's own weight), so this fusion needs no cross-component plumbing. See
  // the comment above x_normed's own epilogue handling for why EpilogueForLayout always returns
  // r4dx_epilogue_none today.
  uint16_t* h = arena.Alloc<uint16_t>(static_cast<size_t>(T * intermediate));
  const int down_epilogue = EpilogueForLayout(w_.down.layout);
  // A trellis mlp.down's A straight from silu_mul (the fused producer below).
  const bool down_trellis_fused = down_had_signs_ == nullptr && TrellisFusionEnabled() &&
                                  w_.down.layout == Layout::kTrellis && w_.down.trellis_parts == 1 &&
                                  w_.down.K == intermediate;
  // R4DX_PREFILL_INT8_FUSEDQ: when mlp.down will take the int8 GEMM (the same test its ApplyLinear makes), the
  // producer writes the int8 operand (A8 + SA) and no f16 A exists.
  const bool down_trellis_i8 = down_trellis_fused && TrellisI8FusedQ(w_.down, T);
  uint16_t* down_trellis_a = nullptr;
  TrellisI8Operand down_i8;
  if (down_trellis_i8) {
    down_i8 = AllocTrellisI8Operand(arena, intermediate, 1);
  } else if (down_trellis_fused) {
    down_trellis_a =
        arena.Alloc<uint16_t>(static_cast<size_t>(T * intermediate), /*align_bytes=*/16);
  }
  uint8_t* down_pre_data = nullptr;
  if (down_epilogue != r4dx_epilogue_none) {
    // f16: 2 bytes per element. 16-byte alignment: see attention_layer.hpp's identical comment.
    down_pre_data =
        arena.Alloc<uint8_t>(static_cast<size_t>(T * intermediate * 2), /*align_bytes=*/16);
  }
  if (down_had_signs_ != nullptr) {
    // quant2 Q2b (docs/quant2.md section 4): h = (silu(gate) * up) Hb in one launch -- the product
    // stays fp32 into the transform, and the epilogue (if any) casts the ROTATED row, which is
    // what the W Hb-folded mlp.down expects.
    ProfiledCall(prof, s_raw, "mlp.silu_mul_hadamard", [&] {
      r4dx_silu_mul_hadamard_bf16(reinterpret_cast<int64_t>(gate_up), reinterpret_cast<int64_t>(h),
                                   T, intermediate, /*in_row_stride=*/2 * intermediate, s,
                                   down_epilogue, reinterpret_cast<int64_t>(down_pre_data),
                                   reinterpret_cast<int64_t>(down_had_signs_),
                                   static_cast<int>(kHadDownBlock));
    });
  } else if (down_trellis_fused) {
    // docs/trellis-kernel.md 4.8 / 5.4 (M5): the product straight into mlp.down's input transform
    // (r4dx_silu_mul_trellis_bf16: the same bf16 h as the wide kernel below, then the same
    // transform ApplyLinear would run, so the same bytes) -- one launch; `h` is never written.
    ProfiledCall(prof, s_raw, "mlp.silu_mul_trellis", [&] {
      if (down_trellis_i8) {
        // ... and, on the int8 path, straight on into the quantizer: A8 + SA are the bytes
        // r4d_trellis_i8_quant_act would make from that f16 A.
        r4dx_silu_mul_trellis_i8(reinterpret_cast<int64_t>(gate_up), T, intermediate,
                                 /*in_row_stride=*/2 * intermediate,
                                 reinterpret_cast<int64_t>(w_.down.trellis_suh.data()),
                                 reinterpret_cast<int64_t>(down_i8.a8), reinterpret_cast<int64_t>(down_i8.sa),
                                 w_.down.trellis_prescale_log2, s);
        return;
      }
      r4dx_silu_mul_trellis_bf16(reinterpret_cast<int64_t>(gate_up), T, intermediate,
                                 /*in_row_stride=*/2 * intermediate,
                                 reinterpret_cast<int64_t>(w_.down.trellis_suh.data()),
                                 reinterpret_cast<int64_t>(down_trellis_a),
                                 w_.down.trellis_prescale_log2, s);
    });
  } else if (w_.down.layout == Layout::kTrellis && down_epilogue == r4dx_epilogue_none) {
    // docs/trellis-kernel.md 10.2: a trellis mlp.down takes plain bf16 h (EpilogueForLayout is
    // none; ApplyLinear runs its input transform), so no epilogue is fused here -- the wide-grid
    // form, the same bytes as r4dx_silu_mul_bf16 without the one-workgroup-per-row decode cost.
    ProfiledCall(prof, s_raw, "mlp.silu_mul", [&] {
      r4dx_silu_mul_wide_bf16(reinterpret_cast<int64_t>(gate_up), reinterpret_cast<int64_t>(h), T,
                              intermediate, /*in_row_stride=*/2 * intermediate, s);
    });
  } else {
    ProfiledCall(prof, s_raw, "mlp.silu_mul", [&] {
      r4dx_silu_mul_bf16(reinterpret_cast<int64_t>(gate_up), reinterpret_cast<int64_t>(h), T,
                          intermediate, /*in_row_stride=*/2 * intermediate, s, down_epilogue,
                          reinterpret_cast<int64_t>(down_pre_data));
    });
  }

  uint16_t* down_out = arena.Alloc<uint16_t>(static_cast<size_t>(T * hidden));
  ProfiledCall(prof, s_raw, "gemm:mlp.down", [&] {
    PreQuantizedActivation pre{down_epilogue, down_pre_data};
    if (down_trellis_fused) {
      pre = PreQuantizedActivation{r4dx_epilogue_none, down_trellis_a, w_.down.trellis_suh.data(),
                                   0};
      if (down_trellis_i8) {
        pre.a8 = down_i8.a8;
        pre.sa = down_i8.sa;
      }
    }
    ApplyLinear(stream, arena, w_.down, h, down_out, T,
                down_trellis_fused || down_epilogue != r4dx_epilogue_none ? &pre : nullptr);
  });
  // Tensor parallel (docs/tp.md 6.2, site A3): down is row-parallel, so each rank holds a partial
  // sum; sum it across ranks before the residual (and the fused next-norm epilogue).
  if (comm_ != nullptr) {
    ProfiledCall(prof, s_raw, "tp.allreduce",
                 [&] { comm_->AllReduceSumBf16Rows(down_out, T, hidden, s_raw); });
  }

  ProfiledCall(prof, s_raw, "mlp.residual", [&] {
    if (next_norm_weight != nullptr) {
      r4dx_residual_rmsnorm_bf16(reinterpret_cast<int64_t>(x), reinterpret_cast<int64_t>(down_out),
                                  reinterpret_cast<int64_t>(next_norm_weight),
                                  reinterpret_cast<int64_t>(x_out),
                                  reinterpret_cast<int64_t>(x_normed_out), T, hidden, eps, s,
                                  next_epilogue, reinterpret_cast<int64_t>(next_epilogue_out));
    } else {
      r4dx_residual_add_bf16(reinterpret_cast<int64_t>(x), reinterpret_cast<int64_t>(down_out),
                              reinterpret_cast<int64_t>(x_out), T * hidden, s);
    }
  });
}

}  // namespace r4dx::model
