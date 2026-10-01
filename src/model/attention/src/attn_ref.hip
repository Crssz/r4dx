// src/model/attention/src/attn_ref.hip -- implements include/r4dx/model/attention/attn_ref.h, the
// stage-0 reference attention for Gemma 4 (docs/gemma4-plan.md 3.3). Compiled by hipcc with the
// other r4dx_model_attention objects (src/model/attention/CMakeLists.txt), -ffp-contract=off.
//
// One workgroup (256 threads) per (query row t, q head h). Three phases over the visible key range
// [klow, khigh] (contiguous, so there is no mask to apply -- the range IS the mask):
//   1. max of the scores (a thread owns one key of each 256-key chunk and computes its dot product);
//   2. the same scores again, p = exp(s - m), staged through LDS in chunks of 256 so that thread d
//      can accumulate sum_j p_j * V[j][d] over its own output columns; the row sum l alongside;
//   3. out[d] = acc[d] / l * v_descale, rounded once to bf16.
// Scores are recomputed rather than stored, so no LDS grows with the context: the kernel is exact at
// any length and as slow as that sounds. Every sum is in a fixed order (sequential d per key,
// ascending key within a column, a fixed block reduction), so a launch is deterministic.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

#include "r4d_attn_window.h"
#include "r4dx/core/dtype.hpp"
#include "r4dx/model/attention/attn_ref.h"

namespace {

constexpr int kThreads = 256;
constexpr int kMaxHeadDim = 512;   // two output columns per thread

__device__ __forceinline__ float BlockReduce(float v, bool is_max, float* red) {
  for (int o = 16; o; o >>= 1) {
    const float w = __shfl_xor(v, o, 32);
    v = is_max ? fmaxf(v, w) : v + w;
  }
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float out = red[0];
  for (int i = 1; i < kThreads / 32; ++i) out = is_max ? fmaxf(out, red[i]) : out + red[i];
  __syncthreads();   // red is reused by the next reduction
  return out;
}

template <int KVP>
__device__ __forceinline__ float LoadKv(const void* kv, int64_t idx) {
  if (KVP) return r4dx::core::Bf16ToFloat(static_cast<const uint16_t*>(kv)[idx]);
  return r4dx::core::Fp8E4M3ToFloat(static_cast<const uint8_t*>(kv)[idx]);
}

template <int KVP>
__global__ __launch_bounds__(kThreads) void GemmaAttnRefKernel(const R4dxGemmaAttnRefArgs a) {
  __shared__ float sq[kMaxHeadDim];
  __shared__ float sp[kThreads];
  __shared__ float red[kThreads / 32];

  const int h = blockIdx.x, t = blockIdx.y;
  const int hd = a.head_dim, bs = a.block_size;
  const int kvh = h / (a.q_heads / a.kv_heads);
  const int qpos = a.ctx - a.q_len + t;
  const int* bt = reinterpret_cast<const int*>(a.block_table);
  const void* kv = reinterpret_cast<const void*>(a.kv);

  const int ext = a.klimit_ext ? reinterpret_cast<const int*>(a.klimit_ext)[t] : -1;
  const int klow = r4d_win_klow(qpos, a.window);
  int khigh = r4d_win_khigh(qpos, ext);
  if (khigh > a.ctx - 1) khigh = a.ctx - 1;

  const float kdesc = a.k_descale ? reinterpret_cast<const float*>(a.k_descale)[kvh] : 1.0f;
  const float vdesc = a.v_descale ? reinterpret_cast<const float*>(a.v_descale)[kvh] : 1.0f;
  const float sscale = a.scale * kdesc;

  const uint16_t* qrow = reinterpret_cast<const uint16_t*>(a.q) +
                         (static_cast<int64_t>(t) * a.q_heads + h) * hd;
  for (int d = threadIdx.x; d < hd; d += kThreads) sq[d] = r4dx::core::Bf16ToFloat(qrow[d]);
  __syncthreads();

  // Element offset of key k's row (K columns first, V at + head_dim).
  auto key_base = [&](int k) -> int64_t {
    return static_cast<int64_t>(bt[k / bs]) * a.kv_block_stride +
           static_cast<int64_t>(kvh) * a.kv_head_stride + static_cast<int64_t>(k % bs) * 2 * hd;
  };
  auto score = [&](int k) -> float {
    const int64_t base = key_base(k);
    float acc = 0.0f;
    for (int d = 0; d < hd; ++d) acc += sq[d] * LoadKv<KVP>(kv, base + d);
    return acc * sscale;
  };

  float m = -INFINITY;
  for (int k0 = klow; k0 <= khigh; k0 += kThreads) {
    const int k = k0 + static_cast<int>(threadIdx.x);
    if (k <= khigh) m = fmaxf(m, score(k));
  }
  m = BlockReduce(m, true, red);

  float l = 0.0f, acc0 = 0.0f, acc1 = 0.0f;
  for (int k0 = klow; k0 <= khigh; k0 += kThreads) {
    const int k = k0 + static_cast<int>(threadIdx.x);
    float p = 0.0f;
    if (k <= khigh) p = expf(score(k) - m);
    sp[threadIdx.x] = p;
    l += p;
    __syncthreads();
    const int n = (khigh - k0 + 1 < kThreads) ? (khigh - k0 + 1) : kThreads;
    const int d0 = threadIdx.x, d1 = threadIdx.x + kThreads;
    for (int j = 0; j < n; ++j) {
      const int64_t vb = key_base(k0 + j) + hd;
      const float pj = sp[j];
      acc0 += pj * LoadKv<KVP>(kv, vb + d0);
      if (d1 < hd) acc1 += pj * LoadKv<KVP>(kv, vb + d1);
    }
    __syncthreads();
  }
  l = BlockReduce(l, false, red);

  const float inv = (l > 0.0f) ? (vdesc / l) : 0.0f;
  uint16_t* orow = reinterpret_cast<uint16_t*>(a.out) + (static_cast<int64_t>(t) * a.q_heads + h) * hd;
  if (threadIdx.x < hd) orow[threadIdx.x] = r4dx::core::FloatToBf16(acc0 * inv);
  if (threadIdx.x + kThreads < hd) orow[threadIdx.x + kThreads] = r4dx::core::FloatToBf16(acc1 * inv);
}

template <int KVP>
int Launch(const R4dxGemmaAttnRefArgs* a, int64_t stream) {
  if (a == nullptr) return -1;
  if (a->q_len <= 0 || a->q_heads <= 0 || a->kv_heads <= 0 || a->block_size <= 0 || a->ctx <= 0 ||
      a->q_len > a->ctx || a->window < 0) {
    return -1;
  }
  if (a->head_dim < 8 || a->head_dim > kMaxHeadDim || (a->head_dim % 8) != 0) return -2;
  if (a->q_heads % a->kv_heads != 0) return -3;
  if (a->q == 0 || a->kv == 0 || a->block_table == 0 || a->out == 0) return -4;
  hipLaunchKernelGGL(GemmaAttnRefKernel<KVP>,
                     dim3(static_cast<unsigned>(a->q_heads), static_cast<unsigned>(a->q_len)),
                     dim3(kThreads), 0, reinterpret_cast<hipStream_t>(stream), *a);
  const hipError_t e = hipGetLastError();
  return e == hipSuccess ? 0 : -100 - static_cast<int>(e);
}

}  // namespace

extern "C" int r4dx_gemma_attn_ref_fp8kv(const R4dxGemmaAttnRefArgs* a, int64_t stream) {
  return Launch<0>(a, stream);
}
extern "C" int r4dx_gemma_attn_ref_bf16kv(const R4dxGemmaAttnRefArgs* a, int64_t stream) {
  return Launch<1>(a, stream);
}
