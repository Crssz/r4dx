// rep_penalty.hip -- r4dx_apply_repetition_penalty_f32 (include/r4dx/kernels/rep_penalty.h; docs/lfm2-plan.md
// 5.4). Its own hipcc object (src/kernels/CMakeLists.txt, appended through target_sources), built with
// exactly r4dx_kernels.hip's flags: -ffp-contract=off, IEEE fp32 divide. One thread per seen id, ids are
// distinct so there are no write conflicts.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "r4dx/core/error.hpp"
#include "r4dx/kernels/kernels.h"  // r4dx_kernel_launch_counter_add
#include "r4dx/kernels/rep_penalty.h"

namespace {

constexpr int kThreads = 256;

__global__ __launch_bounds__(kThreads) void RepetitionPenaltyKernel(float* __restrict__ logits,
                                                                    const int32_t* __restrict__ ids,
                                                                    int64_t n_seen, int64_t vocab, float p) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n_seen) return;
  const int32_t id = ids[i];
  if (id < 0 || id >= vocab) return;
  const float v = logits[id];
  logits[id] = v < 0.0f ? v * p : v / p;
}

}  // namespace

extern "C" void r4dx_apply_repetition_penalty_f32(int64_t logits, int64_t seen_ids, int64_t n_seen,
                                                   int64_t vocab, float penalty, int64_t stream) {
  if (n_seen <= 0 || penalty == 1.0f) return;
  if (!(penalty > 0.0f) || !std::isfinite(penalty)) {
    throw std::runtime_error("r4dx_apply_repetition_penalty_f32: penalty must be finite and > 0");
  }
  if (vocab <= 0) throw std::runtime_error("r4dx_apply_repetition_penalty_f32: vocab must be > 0");
  if (logits == 0 || seen_ids == 0) {
    throw std::runtime_error("r4dx_apply_repetition_penalty_f32: logits and seen_ids must be non-null");
  }
  r4dx_kernel_launch_counter_add(1);
  const unsigned blocks = static_cast<unsigned>((n_seen + kThreads - 1) / kThreads);
  hipLaunchKernelGGL(RepetitionPenaltyKernel, dim3(blocks), dim3(kThreads), 0, (hipStream_t)stream,
                     (float*)logits, (const int32_t*)seen_ids, n_seen, vocab, penalty);
  R4DX_HIP_CHECK(hipGetLastError());
}