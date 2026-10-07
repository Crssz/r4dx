#include "final_lm_head.h"

#include "kernels/model_kernels.h"
#include "linear.h"
#include "profile_span.h"
#include "r4dx/kernels/kernels.h"

namespace r4dx::model {

const uint16_t* FinalLmHead::ForwardBf16(core::Stream& stream, core::Arena& arena, const uint16_t* x,
                                          int64_t T) {
  const int64_t hidden = cfg_.hidden_size;
  const int64_t vocab = lm_head_.N;
  const float eps = static_cast<float>(cfg_.rms_norm_eps);
  const int64_t s = reinterpret_cast<int64_t>(stream.get());

  uint16_t* x_normed = arena.Alloc<uint16_t>(static_cast<size_t>(T * hidden));
  r4dx_rmsnorm_bf16(reinterpret_cast<int64_t>(x), reinterpret_cast<int64_t>(final_norm_.data()),
                     reinterpret_cast<int64_t>(x_normed), T, hidden, eps, s);

  uint16_t* logits_bf16 = arena.Alloc<uint16_t>(static_cast<size_t>(T * vocab));
  // A span of its own only for R4DX_PROFILE_LINEARS (debug_probe.h); callers that profile with
  // hipEvents wrap this whole method themselves.
  ProfiledCall(nullptr, stream.get(), "gemm:lm_head",
               [&] { ApplyLinear(stream, arena, lm_head_, x_normed, logits_bf16, T); });
  return logits_bf16;
}

void FinalLmHead::Forward(core::Stream& stream, core::Arena& arena, const uint16_t* x,
                           float* logits_out, int64_t T) {
  const int64_t vocab = lm_head_.N;
  const uint16_t* logits_bf16 = ForwardBf16(stream, arena, x, T);
  r4dx_model_widen_bf16_to_f32(reinterpret_cast<int64_t>(logits_bf16),
                                reinterpret_cast<int64_t>(logits_out), T * vocab,
                                reinterpret_cast<int64_t>(stream.get()));
}

}  // namespace r4dx::model
