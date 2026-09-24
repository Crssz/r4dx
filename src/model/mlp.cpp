#include "mlp.h"

#include "linear.h"
#include "profile_span.h"
#include "r4dx/core/tp_comm.hpp"
#include "r4dx/kernels/kernels.h"

namespace r4dx::model {

void Mlp::Forward(core::Stream& stream, core::Arena& arena, const uint16_t* x, uint16_t* x_out,
                   int64_t T, const uint16_t* x_normed_in, const uint16_t* next_norm_weight,
                   uint16_t* x_normed_out, SpanAccumulator* prof, int x_normed_pre_epilogue,
                   const void* x_normed_pre_data, const float* x_normed_pre_scale,
                   int next_epilogue, void* next_epilogue_out, float* next_epilogue_scale) {
  const int64_t hidden = cfg_.hidden_size;
  const int64_t intermediate = cfg_.intermediate_size;
  const float eps = static_cast<float>(cfg_.rms_norm_eps);
  const int64_t s = reinterpret_cast<int64_t>(stream.get());
  const hipStream_t s_raw = stream.get();

  // R3 (docs/r9700.md): skip this block's own rmsnorm launch when the producer already fused it
  // into its residual-add epilogue (r4dx_residual_rmsnorm_bf16) and handed us the result.
  // R2/P2 (docs/r9700.md, 2026-09-20): fused activation-quant epilogues (kernels.h's
  // r4dx_epilogue) are implemented and byte-exact-tested in isolation
  // (tests/kernels/test_fused_quant.cpp, 135/135 checks) and mechanically wired through
  // GdnLayer::Forward/AttentionLayer::Forward/Mlp::Forward/ApplyLinear -- but wiring them into the
  // real model (EpilogueForLayout returning non-NONE) was found, on real hardware, to change the
  // generated token stream for w4a8/mxfp4 even though every individual fused GEMM's output was
  // verified byte-identical to its unfused equivalent by an in-model diagnostic (per-call
  // fused-vs-unfused comparison of gdn.in_proj_qkv/in_proj_z/mlp.down, all showed 0/N differing
  // elements across 200+ real decode-step samples). The root cause was not isolated within this
  // pass's time budget -- see docs/status.md's R2/P2 section for the full incident writeup and
  // recommended next steps. EpilogueForLayout (linear.cpp) therefore returns r4dx_epilogue_none
  // for every layout, which makes x_normed_pre_epilogue/next_epilogue always 0 here too, so the
  // code below always takes the plain (pre-existing, verified-correct) unfused path; it is left in
  // place, structurally wired end-to-end, so a future pass that finds the root cause only has to
  // fix it and flip EpilogueForLayout, not rebuild this plumbing.
  const uint16_t* x_normed = x_normed_in;
  uint16_t* x_normed_scratch = nullptr;
  int gate_up_epilogue = x_normed_pre_epilogue;
  const void* gate_up_pre_data = x_normed_pre_data;
  const float* gate_up_pre_scale = x_normed_pre_scale;
  // Cross-boundary reuse only valid if it matches THIS Mlp's own w_.gate_up layout -- see
  // gdn_layer.cpp's identical defensive comment.
  if (x_normed != nullptr && gate_up_epilogue != EpilogueForLayout(w_.gate_up.layout)) {
    gate_up_epilogue = r4dx_epilogue_none;
    gate_up_pre_data = nullptr;
    gate_up_pre_scale = nullptr;
  }
  if (x_normed == nullptr) {
    gate_up_epilogue = EpilogueForLayout(w_.gate_up.layout);
    x_normed_scratch = arena.Alloc<uint16_t>(static_cast<size_t>(T * hidden));
    uint8_t* gate_up_pre_data_local = nullptr;
    float* gate_up_pre_scale_local = nullptr;
    if (gate_up_epilogue != r4dx_epilogue_none) {
      const int elem_size = (gate_up_epilogue == r4dx_epilogue_f16) ? 2 : 1;
      // 16-byte alignment: see attention_layer.hpp's identical comment.
      gate_up_pre_data_local =
          arena.Alloc<uint8_t>(static_cast<size_t>(T * hidden * elem_size), /*align_bytes=*/16);
      if (gate_up_epilogue != r4dx_epilogue_f16) {
        gate_up_pre_scale_local = arena.Alloc<float>(static_cast<size_t>(T));
      }
    }
    ProfiledCall(prof, s_raw, "mlp.rmsnorm", [&] {
      r4dx_rmsnorm_bf16(reinterpret_cast<int64_t>(x),
                         reinterpret_cast<int64_t>(post_attention_layernorm_.data()),
                         reinterpret_cast<int64_t>(x_normed_scratch), T, hidden, eps, s,
                         gate_up_epilogue, reinterpret_cast<int64_t>(gate_up_pre_data_local),
                         reinterpret_cast<int64_t>(gate_up_pre_scale_local));
    });
    x_normed = x_normed_scratch;
    gate_up_pre_data = gate_up_pre_data_local;
    gate_up_pre_scale = gate_up_pre_scale_local;
  }

  uint16_t* gate_up = arena.Alloc<uint16_t>(static_cast<size_t>(T * 2 * intermediate));
  ProfiledCall(prof, s_raw, "gemm:mlp.gate_up", [&] {
    PreQuantizedActivation pre{gate_up_epilogue, gate_up_pre_data, gate_up_pre_scale};
    ApplyLinear(stream, arena, w_.gate_up, x_normed, gate_up, T,
                gate_up_epilogue != r4dx_epilogue_none ? &pre : nullptr);
  });

  // R2/P2 (docs/r9700.md): silu_mul already touches every element of its output row, so it emits
  // w_.down's own required quant format directly (fused epilogue, kernels.h's r4dx_epilogue),
  // letting ApplyLinear below skip its separate quant/cast launch for this GEMM. w_.down's layout
  // is known locally (this Mlp's own weight), so this fusion needs no cross-component plumbing.
  // See the comment above x_normed's own epilogue handling for why EpilogueForLayout always
  // returns r4dx_epilogue_none today (unresolved full-model correctness issue, not yet fixed).
  uint16_t* h = arena.Alloc<uint16_t>(static_cast<size_t>(T * intermediate));
  const int down_epilogue = EpilogueForLayout(w_.down.layout);
  uint8_t* down_pre_data = nullptr;
  float* down_pre_scale = nullptr;
  if (down_epilogue != r4dx_epilogue_none) {
    const int elem_size = (down_epilogue == r4dx_epilogue_f16) ? 2 : 1;
    // 16-byte alignment: see attention_layer.hpp's identical comment.
    down_pre_data =
        arena.Alloc<uint8_t>(static_cast<size_t>(T * intermediate * elem_size), /*align_bytes=*/16);
    if (down_epilogue != r4dx_epilogue_f16) {
      down_pre_scale = arena.Alloc<float>(static_cast<size_t>(T));
    }
  }
  ProfiledCall(prof, s_raw, "mlp.silu_mul", [&] {
    r4dx_silu_mul_bf16(reinterpret_cast<int64_t>(gate_up), reinterpret_cast<int64_t>(h), T,
                        intermediate, /*in_row_stride=*/2 * intermediate, s, down_epilogue,
                        reinterpret_cast<int64_t>(down_pre_data),
                        reinterpret_cast<int64_t>(down_pre_scale));
  });

  uint16_t* down_out = arena.Alloc<uint16_t>(static_cast<size_t>(T * hidden));
  ProfiledCall(prof, s_raw, "gemm:mlp.down", [&] {
    PreQuantizedActivation pre{down_epilogue, down_pre_data, down_pre_scale};
    ApplyLinear(stream, arena, w_.down, h, down_out, T,
                down_epilogue != r4dx_epilogue_none ? &pre : nullptr);
  });
  // Tensor parallel (docs/tp.md 6.2, site A3): down is row-parallel, so each rank holds a partial
  // sum; sum it across ranks before the residual (and the fused next-norm epilogue).
  if (comm_ != nullptr) {
    ProfiledCall(prof, s_raw, "tp.allreduce",
                 [&] { comm_->AllReduceSumBf16(down_out, T * hidden, s_raw); });
  }

  ProfiledCall(prof, s_raw, "mlp.residual", [&] {
    if (next_norm_weight != nullptr) {
      r4dx_residual_rmsnorm_bf16(reinterpret_cast<int64_t>(x), reinterpret_cast<int64_t>(down_out),
                                  reinterpret_cast<int64_t>(next_norm_weight),
                                  reinterpret_cast<int64_t>(x_out),
                                  reinterpret_cast<int64_t>(x_normed_out), T, hidden, eps, s,
                                  next_epilogue, reinterpret_cast<int64_t>(next_epilogue_out),
                                  reinterpret_cast<int64_t>(next_epilogue_scale));
    } else {
      r4dx_residual_add_bf16(reinterpret_cast<int64_t>(x), reinterpret_cast<int64_t>(down_out),
                              reinterpret_cast<int64_t>(x_out), T * hidden, s);
    }
  });
}

}  // namespace r4dx::model
