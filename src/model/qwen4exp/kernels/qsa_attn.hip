// QSA layer, stage 3: attention of every query over its selected tokens (or [0, p] for dense rows) from the fp8
// paged KV cache, plus the output gate. Contracts and rounding points: include/r4dx/model/qwen4exp/qsa_kernels.h.
//
// Shape: gqa 12 (one kv head serves 12 query heads), head_dim 256, PagedKvCache block 16, K row then V row of 256
// fp8 e4m3 bytes per (block, kv head, slot). Flash-decoding structure:
//   AttnPartialKernel  grid (nsplit, kv heads, T), 256 threads. A workgroup walks `tiles_per_split` tiles of 256
//     tokens of one (query, kv head). Per tile, thread j owns token j: it dots the 12 q heads (q in LDS, fp32)
//     with the token's fp8 K row (hardware fp8 -> fp32 convert) -> scores in LDS; the tile max per head and the
//     online-softmax rescale are uniform over the workgroup; then thread d owns head dim d and accumulates
//     p[token][head] * V[token][d] for the 12 heads (p from LDS, V one byte per thread per token, coalesced).
//     The unnormalised accumulator, running max and sum of the split go to the partial buffers.
//   AttnCombineKernel  grid (heads_q, T): merges the splits (max-shifted), multiplies by v_descale / l, rounds
//     to bf16, multiplies by bf16(sigmoid(gate)) and rounds again (HF: `attn_output * torch.sigmoid(gate)`).
// Token source: dense rows (nb = (p + 1) / ratio <= topk) read tokens 0..p; sparse rows read the selection
// kernel's ascending list. n_tok <= budget + ratio - 1 either way.
#include <hip/hip_runtime.h>

#include "r4dx/model/qwen4exp/qsa_device.h"
#include "r4dx/model/qwen4exp/qsa_kernels.h"

namespace qsa = r4dx::model::qwen4exp::qsa;
using qsa::Bf16ToFloat;
using qsa::FloatToBf16;

namespace {

constexpr int kG = 12;        // query heads per kv head
constexpr int kHD = 256;      // head_dim
constexpr int kTile = 256;    // tokens per tile = threads
constexpr int kWaves = kTile / 32;
constexpr int kBlockSize = 16;  // PagedKvCache slots per block
constexpr float kScale = 0.0625f;  // 1 / sqrt(256), exact

typedef float v2f __attribute__((ext_vector_type(2)));

__global__ __launch_bounds__(kTile) void AttnPartialKernel(Q4QsaAttnArgs a) {
  __shared__ __attribute__((aligned(16))) float q_lds[kHD * kG];
  __shared__ __attribute__((aligned(16))) float s_lds[kTile * kG];
  __shared__ unsigned long long off_lds[kTile];
  __shared__ float red[kWaves * kG];
  __shared__ float m_tile[kG];

  const int sp = blockIdx.x, g = blockIdx.y, t = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31, wave = tid >> 5;
  const int p = a.positions[t];
  const bool sparse = (p + 1) / a.ratio > a.topk;
  const int n_tok = sparse ? a.sel_count[t] : p + 1;
  const int ntiles = (n_tok + kTile - 1) / kTile;
  const int tile0 = sp * a.tiles_per_split;
  if (tile0 >= ntiles) return;  // empty split: the combine kernel never reads it
  const int tile1 = min(ntiles, tile0 + a.tiles_per_split);

  {
    const uint16_t* qsrc = a.q + (static_cast<int64_t>(t) * a.heads_q + g * kG) * kHD;
    for (int idx = tid; idx < kG * kHD; idx += kTile) {
      q_lds[(idx % kHD) * kG + idx / kHD] = Bf16ToFloat(qsrc[idx]);
    }
  }
  __syncthreads();
  const float k_scale = a.k_descale[g] * kScale;
  const int32_t* list = a.sel_tokens + static_cast<int64_t>(t) * a.sel_stride;
  const uint8_t* head_base = a.kv_cache + static_cast<int64_t>(g) * a.kv_head_stride;

  float acc[kG], lp[kG], m_run[kG];
#pragma unroll
  for (int h = 0; h < kG; ++h) {
    acc[h] = 0.0f;
    lp[h] = 0.0f;
    m_run[h] = -INFINITY;
  }

  for (int tile = tile0; tile < tile1; ++tile) {
    const int i = tile * kTile + tid;
    const bool valid = i < n_tok;
    float s[kG];
#pragma unroll
    for (int h = 0; h < kG; ++h) s[h] = 0.0f;
    unsigned long long off = 0;
    if (valid) {
      const int tok = sparse ? list[i] : i;
      off = static_cast<unsigned long long>(a.block_table[tok / kBlockSize]) * a.kv_block_stride +
            static_cast<unsigned long long>(tok % kBlockSize) * (2 * kHD);
      const uint4* krow = reinterpret_cast<const uint4*>(head_base + off);
#pragma unroll 1
      for (int c = 0; c < kHD / 16; ++c) {
        const uint4 raw = krow[c];
        const unsigned w[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const v2f lo = __builtin_amdgcn_cvt_pk_f32_fp8(w[j], false);
          const v2f hi = __builtin_amdgcn_cvt_pk_f32_fp8(w[j], true);
          const float x[4] = {lo.x, lo.y, hi.x, hi.y};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float4* qp = reinterpret_cast<const float4*>(&q_lds[(c * 16 + j * 4 + e) * kG]);
            const float4 q0 = qp[0], q1 = qp[1], q2 = qp[2];
            s[0] += x[e] * q0.x;  s[1] += x[e] * q0.y;  s[2] += x[e] * q0.z;   s[3] += x[e] * q0.w;
            s[4] += x[e] * q1.x;  s[5] += x[e] * q1.y;  s[6] += x[e] * q1.z;   s[7] += x[e] * q1.w;
            s[8] += x[e] * q2.x;  s[9] += x[e] * q2.y;  s[10] += x[e] * q2.z;  s[11] += x[e] * q2.w;
          }
        }
      }
    }
#pragma unroll
    for (int h = 0; h < kG; ++h) s[h] = valid ? s[h] * k_scale : -INFINITY;
    off_lds[tid] = off;

    // tile max per head
#pragma unroll
    for (int h = 0; h < kG; ++h) {
      const float v = qsa::WaveMax(s[h]);
      if (lane == 0) red[wave * kG + h] = v;
    }
    __syncthreads();
    if (tid < kG) {
      float m = -INFINITY;
#pragma unroll
      for (int w = 0; w < kWaves; ++w) m = fmaxf(m, red[w * kG + tid]);
      m_tile[tid] = m;
    }
    __syncthreads();

    float alpha[kG], pr[kG];
#pragma unroll
    for (int h = 0; h < kG; ++h) {
      const float m_new = fmaxf(m_run[h], m_tile[h]);
      alpha[h] = expf(m_run[h] - m_new);  // 0 on the first tile (m_run = -inf)
      m_run[h] = m_new;
      pr[h] = valid ? expf(s[h] - m_new) : 0.0f;
      lp[h] = lp[h] * alpha[h] + pr[h];
      acc[h] *= alpha[h];
    }
    float4* sp4 = reinterpret_cast<float4*>(&s_lds[tid * kG]);
    sp4[0] = make_float4(pr[0], pr[1], pr[2], pr[3]);
    sp4[1] = make_float4(pr[4], pr[5], pr[6], pr[7]);
    sp4[2] = make_float4(pr[8], pr[9], pr[10], pr[11]);
    __syncthreads();

    const int nt = min(kTile, n_tok - tile * kTile);
    const uint8_t* vcol = head_base + kHD + tid;
#pragma unroll 4
    for (int j = 0; j < nt; ++j) {
      const unsigned vb = vcol[off_lds[j]];
      const float vf = __builtin_amdgcn_cvt_f32_fp8(vb, 0);
      const float4* pp = reinterpret_cast<const float4*>(&s_lds[j * kG]);
      const float4 p0 = pp[0], p1 = pp[1], p2 = pp[2];
      acc[0] += p0.x * vf;  acc[1] += p0.y * vf;  acc[2] += p0.z * vf;   acc[3] += p0.w * vf;
      acc[4] += p1.x * vf;  acc[5] += p1.y * vf;  acc[6] += p1.z * vf;   acc[7] += p1.w * vf;
      acc[8] += p2.x * vf;  acc[9] += p2.y * vf;  acc[10] += p2.z * vf;  acc[11] += p2.w * vf;
    }
    __syncthreads();  // s_lds / off_lds / red are rewritten by the next tile
  }

  // sum of the per-thread partial sums
#pragma unroll
  for (int h = 0; h < kG; ++h) {
    const float v = qsa::WaveSum(lp[h]);
    if (lane == 0) red[wave * kG + h] = v;
  }
  __syncthreads();
  const int64_t slot = (static_cast<int64_t>(t) * a.heads_kv + g) * a.nsplit + sp;
  if (tid < kG) {
    float l = 0.0f;
#pragma unroll
    for (int w = 0; w < kWaves; ++w) l += red[w * kG + tid];
    a.part_ml[(slot * kG + tid) * 2 + 0] = m_run[tid];
    a.part_ml[(slot * kG + tid) * 2 + 1] = l;
  }
#pragma unroll
  for (int h = 0; h < kG; ++h) a.part_acc[(slot * kG + h) * kHD + tid] = acc[h];
}

__global__ __launch_bounds__(kHD) void AttnCombineKernel(Q4QsaAttnArgs a) {
  const int h = blockIdx.x, t = blockIdx.y;
  const int d = threadIdx.x;
  const int g = h / kG, hh = h % kG;
  const int p = a.positions[t];
  const bool sparse = (p + 1) / a.ratio > a.topk;
  const int n_tok = sparse ? a.sel_count[t] : p + 1;
  const int ntiles = (n_tok + kTile - 1) / kTile;
  const int ns = (ntiles + a.tiles_per_split - 1) / a.tiles_per_split;
  const int64_t base = (static_cast<int64_t>(t) * a.heads_kv + g) * a.nsplit;

  float m = -INFINITY;
  for (int s = 0; s < ns; ++s) m = fmaxf(m, a.part_ml[((base + s) * kG + hh) * 2]);
  float l = 0.0f, o = 0.0f;
  for (int s = 0; s < ns; ++s) {
    const float w = expf(a.part_ml[((base + s) * kG + hh) * 2] - m);
    l += w * a.part_ml[((base + s) * kG + hh) * 2 + 1];
    o += w * a.part_acc[((base + s) * kG + hh) * kHD + d];
  }
  const float attn = qsa::Rb((o / l) * a.v_descale[g]);
  const float gate = Bf16ToFloat(a.qg[(static_cast<int64_t>(t) * a.heads_q + h) * 2 * kHD + kHD + d]);
  const float sg = qsa::Rb(1.0f / (1.0f + expf(-gate)));
  a.out[(static_cast<int64_t>(t) * a.heads_q + h) * kHD + d] = FloatToBf16(attn * sg);
}

}  // namespace

extern "C" hipError_t r4dx_q4_qsa_attn(const Q4QsaAttnArgs* a, hipStream_t stream) {
  if (a->T <= 0) return hipSuccess;
  if (a->heads_kv <= 0 || a->heads_q != a->heads_kv * kG || a->tiles_per_split <= 0 || a->nsplit <= 0) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(AttnPartialKernel,
                     dim3(static_cast<unsigned>(a->nsplit), static_cast<unsigned>(a->heads_kv),
                          static_cast<unsigned>(a->T)),
                     dim3(kTile), 0, stream, *a);
  hipLaunchKernelGGL(AttnCombineKernel, dim3(static_cast<unsigned>(a->heads_q), static_cast<unsigned>(a->T)),
                     dim3(kHD), 0, stream, *a);
  return hipGetLastError();
}
