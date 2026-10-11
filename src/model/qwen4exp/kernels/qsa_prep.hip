// QSA layer, stage 1: the epilogue of the q/k/v/index projections (q/k RMSNorm + rope + fp8 cache write, indexer
// query norm + rope, raw indexer keys) and the indexer block keys. Contracts and rounding points:
// include/r4dx/model/qwen4exp/qsa_kernels.h.
#include <hip/hip_runtime.h>

#include "r4dx/model/qwen4exp/qsa_device.h"
#include "r4dx/model/qwen4exp/qsa_kernels.h"

namespace qsa = r4dx::model::qwen4exp::qsa;
using qsa::Bf16ToFloat;
using qsa::FloatToBf16;

namespace {

constexpr int kMaxRow = 256;  // head_dim

// grid (T, heads_q + heads_kv + index_heads + 1); one wave per row. Row kinds by blockIdx.y:
//   [0, Hq)                       q head: norm, rope -> q_out (the gate half of qg is read later by the attention)
//   [Hq, Hq + Hkv)                k head: norm, rope -> fp8 K cache; the same head's v row -> fp8 V cache
//   next index_heads              indexer query head: q_layernorm, rope -> idx_q_out
//   last                          raw indexer key: copied to raw_keys[position]
// The fp8 rows are fp8(bf16 value * 1 / descale) exactly as r4dx_kv_write_paged_fp8_hnd writes them.
__global__ __launch_bounds__(32) void PrepKernel(Q4QsaPrepArgs a) {
  __shared__ __attribute__((aligned(16))) uint16_t row[kMaxRow];
  const int t = blockIdx.x;
  const int y = blockIdx.y;
  const int lane = threadIdx.x;
  const int pos = a.positions[t];
  const int half = a.rotary_dim / 2;

  if (y < a.heads_q) {
    const uint16_t* src = a.qg + (static_cast<int64_t>(t) * a.heads_q + y) * 2 * a.head_dim;
    for (int i = lane; i < a.head_dim; i += 32) row[i] = src[i];
    __syncthreads();
    qsa::NormRowInPlace<256>(row, a.q_norm, a.eps, lane);
    __syncthreads();
    qsa::RopeRowInPlace(row, a.inv_freq, half, pos, lane);
    __syncthreads();
    uint16_t* dst = a.q_out + (static_cast<int64_t>(t) * a.heads_q + y) * a.head_dim;
    for (int i = lane; i < a.head_dim; i += 32) dst[i] = row[i];
    return;
  }

  const int ky = y - a.heads_q;
  if (ky < a.heads_kv) {
    const uint16_t* src = a.k + (static_cast<int64_t>(t) * a.heads_kv + ky) * a.head_dim;
    for (int i = lane; i < a.head_dim; i += 32) row[i] = src[i];
    __syncthreads();
    qsa::NormRowInPlace<256>(row, a.k_norm, a.eps, lane);
    __syncthreads();
    qsa::RopeRowInPlace(row, a.inv_freq, half, pos, lane);
    __syncthreads();
    if (pos < 0) return;  // -1 = skip the cache write (padded token)
    const int block_id = pos / a.block_size;
    const int offset = pos % a.block_size;
    const int64_t base = static_cast<int64_t>(block_id) * a.kv_block_stride +
                         static_cast<int64_t>(ky) * a.kv_head_stride +
                         static_cast<int64_t>(offset) * (2 * a.head_dim);
    const float kd_inv = 1.0f / a.k_descale[ky];
    const float vd_inv = 1.0f / a.v_descale[ky];
    const uint16_t* vv = a.v + (static_cast<int64_t>(t) * a.heads_kv + ky) * a.head_dim;
    for (int d = lane; d < a.head_dim; d += 32) {
      a.kv_cache[base + d] = r4dx::core::FloatToFp8E4M3(Bf16ToFloat(row[d]) * kd_inv);
      a.kv_cache[base + a.head_dim + d] = r4dx::core::FloatToFp8E4M3(Bf16ToFloat(vv[d]) * vd_inv);
    }
    return;
  }

  const int iy = ky - a.heads_kv;
  const uint16_t* idx = a.idxqk + static_cast<int64_t>(t) * (a.index_heads + 1) * a.index_dim;
  if (iy < a.index_heads) {
    const uint16_t* src = idx + static_cast<int64_t>(iy) * a.index_dim;
    for (int i = lane; i < a.index_dim; i += 32) row[i] = src[i];
    __syncthreads();
    qsa::NormRowInPlace<128>(row, a.idx_q_norm, a.eps, lane);
    __syncthreads();
    qsa::RopeRowInPlace(row, a.inv_freq, half, pos, lane);
    __syncthreads();
    uint16_t* dst = a.idx_q_out + (static_cast<int64_t>(t) * a.index_heads + iy) * a.index_dim;
    for (int i = lane; i < a.index_dim; i += 32) dst[i] = row[i];
    return;
  }

  if (pos < 0) return;
  const uint16_t* src = idx + static_cast<int64_t>(a.index_heads) * a.index_dim;
  uint16_t* dst = a.raw_keys + static_cast<int64_t>(pos) * a.index_dim;
  for (int i = lane; i < a.index_dim; i += 32) dst[i] = src[i];
}

// One wave per completing block: pooled = Rb(mean_fp32(raw keys)), k_layernorm, rope at the block's first position.
__global__ __launch_bounds__(32) void BlockKeyKernel(Q4QsaBlockKeyArgs a) {
  __shared__ __attribute__((aligned(16))) uint16_t row[128];
  const int lane = threadIdx.x;
  const int p0 = a.positions[0];
  const int p_last = a.positions[a.T - 1];
  const int b = p0 / a.ratio + static_cast<int>(blockIdx.x);
  if (b >= (p_last + 1) / a.ratio) return;  // the block does not complete inside this chunk
  const uint16_t* keys = a.raw_keys + static_cast<int64_t>(b) * a.ratio * a.index_dim;
  for (int i = lane; i < a.index_dim; i += 32) {
    float sum = 0.0f;
    for (int r = 0; r < a.ratio; ++r) sum += Bf16ToFloat(keys[static_cast<int64_t>(r) * a.index_dim + i]);
    row[i] = FloatToBf16(sum / static_cast<float>(a.ratio));
  }
  __syncthreads();
  qsa::NormRowInPlace<128>(row, a.idx_k_norm, a.eps, lane);
  __syncthreads();
  qsa::RopeRowInPlace(row, a.inv_freq, a.rotary_dim / 2, b * a.ratio, lane);
  __syncthreads();
  uint16_t* dst = a.block_keys + static_cast<int64_t>(b) * a.index_dim;
  for (int i = lane; i < a.index_dim; i += 32) dst[i] = row[i];
}

}  // namespace

extern "C" hipError_t r4dx_q4_qsa_prep(const Q4QsaPrepArgs* a, hipStream_t stream) {
  if (a->T <= 0) return hipSuccess;
  if (a->head_dim != 256 || a->index_dim != 128 || a->rotary_dim / 2 > 32 || a->rotary_dim % 2 != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 grid(static_cast<unsigned>(a->T),
                  static_cast<unsigned>(a->heads_q + a->heads_kv + a->index_heads + 1));
  hipLaunchKernelGGL(PrepKernel, grid, dim3(32), 0, stream, *a);
  return hipGetLastError();
}

extern "C" hipError_t r4dx_q4_qsa_block_keys(const Q4QsaBlockKeyArgs* a, hipStream_t stream) {
  if (a->T <= 0) return hipSuccess;
  if (a->index_dim != 128 || a->ratio <= 0 || a->rotary_dim / 2 > 32) return hipErrorInvalidValue;
  const unsigned grid = static_cast<unsigned>((a->T + a->ratio - 1) / a->ratio + 1);
  hipLaunchKernelGGL(BlockKeyKernel, dim3(grid), dim3(32), 0, stream, *a);
  return hipGetLastError();
}
