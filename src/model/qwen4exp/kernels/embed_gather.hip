// src/model/qwen4exp/kernels/embed_gather.hip -- q4x_embed_gather_repeat_bf16 (contract in
// include/r4dx/model/qwen4exp/hc_kernels.h).
#include <hip/hip_runtime.h>

#include "r4dx/model/qwen4exp/hc_device.h"
#include "r4dx/model/qwen4exp/hc_kernels.h"

namespace dev = r4dx::model::qwen4exp::dev;

namespace {

constexpr uint16_t kBf16QuietNan = 0x7fc0;

// One workgroup per output row: each thread copies a strided set of channels, writing every repeat.
__global__ __launch_bounds__(dev::kBlock) void EmbedGatherRepeatKernel(
    const uint16_t* __restrict__ table, const int32_t* __restrict__ ids, uint16_t* out, int64_t vocab, int hidden,
    int repeat) {
  const int64_t row = blockIdx.x;
  const int64_t id = ids[row];
  const bool ok = id >= 0 && id < vocab;
  const uint16_t* src = table + (ok ? id : 0) * hidden;
  uint16_t* dst = out + row * static_cast<int64_t>(repeat) * hidden;
  for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
    const uint16_t v = ok ? src[d] : kBf16QuietNan;
    for (int g = 0; g < repeat; ++g) dst[static_cast<int64_t>(g) * hidden + d] = v;
  }
}

}  // namespace

extern "C" int q4x_embed_gather_repeat_bf16(const void* table, const int32_t* ids, void* out, int64_t rows,
                                             int64_t vocab, int hidden, int repeat, void* stream) {
  if (rows <= 0) return 0;
  if (hidden < 1 || repeat < 1 || vocab < 1) return hipErrorInvalidValue;
  hipLaunchKernelGGL(EmbedGatherRepeatKernel, dim3(static_cast<unsigned>(rows)), dim3(dev::kBlock), 0,
                     static_cast<hipStream_t>(stream), static_cast<const uint16_t*>(table), ids,
                     static_cast<uint16_t*>(out), vocab, hidden, repeat);
  return static_cast<int>(hipGetLastError());
}
