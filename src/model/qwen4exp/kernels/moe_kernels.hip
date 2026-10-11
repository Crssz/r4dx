// src/model/qwen4exp/kernels/moe_kernels.hip -- contracts in include/r4dx/model/qwen4exp/moe_kernels.h.
#include <hip/hip_runtime.h>

#include <climits>

#include "r4dx/model/qwen4exp/hc_device.h"
#include "r4dx/model/qwen4exp/moe_kernels.h"

namespace dev = r4dx::model::qwen4exp::dev;
using dev::Bf16ToFloat;
using dev::FloatToBf16;
using dev::Rb;

namespace {

constexpr int kRouteWaves = 4;
constexpr int kMaxExperts = 1024;
constexpr int kMaxTopK = 16;

// One wave per row, kRouteWaves rows per workgroup; the row's logits live in the wave's slice of LDS. Every lane of
// the wave runs the same trip counts, so the cross-lane shuffles are always converged. A lane only ever touches
// the LDS entries i == lane (mod 32), so no barrier is needed once the row is loaded (the load itself is a
// wave-local LDS round trip: a barrier orders it).
__global__ __launch_bounds__(kRouteWaves * dev::kWave) void RouteKernel(
    const uint16_t* __restrict__ logits, int32_t* ids, float* weights, int32_t* counts, int64_t rows, int experts,
    int k, int norm_topk) {
  extern __shared__ float lds[];
  const int lane = threadIdx.x & (dev::kWave - 1), wave = threadIdx.x / dev::kWave;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kRouteWaves + wave;
  const bool valid = row < rows;
  float* l = lds + static_cast<size_t>(wave) * experts;

  for (int i = lane; i < experts; i += dev::kWave) l[i] = valid ? Bf16ToFloat(logits[row * experts + i]) : 0.0f;
  __syncthreads();

  float m = -INFINITY;
  for (int i = lane; i < experts; i += dev::kWave) m = fmaxf(m, l[i]);
#pragma unroll
  for (int o = dev::kWave / 2; o; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, dev::kWave));
  float s = 0.0f;
  for (int i = lane; i < experts; i += dev::kWave) s += expf(l[i] - m);
  s = dev::WaveSum(s);

  int my_id = 0;
  float my_p = 0.0f;
  for (int j = 0; j < k; ++j) {
    float best = -INFINITY;
    int bid = INT_MAX;
    for (int i = lane; i < experts; i += dev::kWave) {  // ascending i: the lowest id among equals wins in-lane
      const float v = l[i];
      if (v > best) {
        best = v;
        bid = i;
      }
    }
#pragma unroll
    for (int o = dev::kWave / 2; o; o >>= 1) {
      const float ov = __shfl_xor(best, o, dev::kWave);
      const int oi = __shfl_xor(bid, o, dev::kWave);
      if (ov > best || (ov == best && oi < bid)) {
        best = ov;
        bid = oi;
      }
    }
    if (lane == j) {
      my_id = bid;
      my_p = expf(best - m) / s;
    }
    if ((bid & (dev::kWave - 1)) == lane) l[bid] = -INFINITY;  // taken (the owner lane writes; nobody else reads it)
  }

  float total = 0.0f;
  for (int j = 0; j < k; ++j) total += __shfl(my_p, j, dev::kWave);
  if (valid && lane < k) {
    const float wgt = norm_topk ? my_p / total : my_p;
    ids[row * k + lane] = my_id;
    weights[row * k + lane] = Rb(wgt);
    if (counts != nullptr) atomicAdd(&counts[my_id], 1);
  }
}

__global__ __launch_bounds__(dev::kBlock) void GatherRowsKernel(const uint16_t* __restrict__ x,
                                                                const int32_t* __restrict__ row_idx, uint16_t* out,
                                                                int hidden) {
  const int64_t p = blockIdx.x;
  const uint16_t* src = x + static_cast<int64_t>(row_idx[p]) * hidden;
  uint16_t* dst = out + p * hidden;
  for (int d = threadIdx.x; d < hidden; d += blockDim.x) dst[d] = src[d];
}

__global__ __launch_bounds__(dev::kBlock) void SiluMulKernel(const uint16_t* __restrict__ gate,
                                                             const uint16_t* __restrict__ up, uint16_t* out,
                                                             int inter, int64_t gate_stride, int64_t up_stride) {
  const int64_t r = blockIdx.x;
  const int i = blockIdx.y * blockDim.x + threadIdx.x;
  if (i >= inter) return;
  const float g = Bf16ToFloat(gate[r * gate_stride + i]);
  const float u = Bf16ToFloat(up[r * up_stride + i]);
  out[r * inter + i] = FloatToBf16(Rb(dev::Silu(g)) * u);
}

__global__ __launch_bounds__(dev::kBlock) void CombineKernel(
    const uint16_t* __restrict__ y_pairs, const int32_t* __restrict__ row_pairs,
    const int32_t* __restrict__ row_w_idx, const float* __restrict__ weights, const uint16_t* __restrict__ shared_y,
    const uint16_t* __restrict__ shared_gate_logit, uint16_t* out, int k, int hidden) {
  const int64_t r = blockIdx.x;
  const int d = blockIdx.y * blockDim.x + threadIdx.x;
  if (d >= hidden) return;
  float acc = 0.0f;
  for (int j = 0; j < k; ++j) {
    const int64_t p = row_pairs[r * k + j];
    acc += Bf16ToFloat(y_pairs[p * hidden + d]) * weights[row_w_idx[r * k + j]];
  }
  if (shared_y != nullptr) {
    const float gate = Rb(dev::Sigmoid(Bf16ToFloat(shared_gate_logit[r])));
    acc += Rb(gate * Bf16ToFloat(shared_y[r * hidden + d]));
  }
  out[r * hidden + d] = FloatToBf16(acc);
}

}  // namespace

extern "C" int q4x_moe_route_bf16(const void* logits, int32_t* ids, float* weights, int32_t* counts, int64_t rows,
                                   int experts, int k, int norm_topk, void* stream) {
  if (rows <= 0) return 0;
  if (experts < 1 || experts > kMaxExperts || k < 1 || k > kMaxTopK || k > experts) return hipErrorInvalidValue;
  const unsigned blocks = static_cast<unsigned>((rows + kRouteWaves - 1) / kRouteWaves);
  const size_t lds = static_cast<size_t>(kRouteWaves) * experts * sizeof(float);
  hipLaunchKernelGGL(RouteKernel, dim3(blocks), dim3(kRouteWaves * dev::kWave), lds, static_cast<hipStream_t>(stream),
                     static_cast<const uint16_t*>(logits), ids, weights, counts, rows, experts, k, norm_topk);
  return static_cast<int>(hipGetLastError());
}

extern "C" int q4x_moe_gather_rows_bf16(const void* x, const int32_t* row_idx, void* out, int64_t n, int hidden,
                                         void* stream) {
  if (n <= 0) return 0;
  hipLaunchKernelGGL(GatherRowsKernel, dim3(static_cast<unsigned>(n)), dim3(dev::kBlock), 0,
                     static_cast<hipStream_t>(stream), static_cast<const uint16_t*>(x), row_idx,
                     static_cast<uint16_t*>(out), hidden);
  return static_cast<int>(hipGetLastError());
}

extern "C" int q4x_moe_silu_mul_bf16(const void* gate, const void* up, void* out, int64_t rows, int inter,
                                      int64_t gate_stride, int64_t up_stride, void* stream) {
  if (rows <= 0) return 0;
  const unsigned chunks = static_cast<unsigned>((inter + dev::kBlock - 1) / dev::kBlock);
  hipLaunchKernelGGL(SiluMulKernel, dim3(static_cast<unsigned>(rows), chunks), dim3(dev::kBlock), 0,
                     static_cast<hipStream_t>(stream), static_cast<const uint16_t*>(gate),
                     static_cast<const uint16_t*>(up), static_cast<uint16_t*>(out), inter, gate_stride, up_stride);
  return static_cast<int>(hipGetLastError());
}

extern "C" int q4x_moe_combine_bf16(const void* y_pairs, const int32_t* row_pairs, const int32_t* row_w_idx,
                                     const float* weights, const void* shared_y, const void* shared_gate_logit,
                                     void* out, int64_t rows, int k, int hidden, void* stream) {
  if (rows <= 0) return 0;
  if ((shared_y == nullptr) != (shared_gate_logit == nullptr)) return hipErrorInvalidValue;
  const unsigned chunks = static_cast<unsigned>((hidden + dev::kBlock - 1) / dev::kBlock);
  hipLaunchKernelGGL(CombineKernel, dim3(static_cast<unsigned>(rows), chunks), dim3(dev::kBlock), 0,
                     static_cast<hipStream_t>(stream), static_cast<const uint16_t*>(y_pairs), row_pairs, row_w_idx,
                     weights, static_cast<const uint16_t*>(shared_y), static_cast<const uint16_t*>(shared_gate_logit),
                     static_cast<uint16_t*>(out), k, hidden);
  return static_cast<int>(hipGetLastError());
}
