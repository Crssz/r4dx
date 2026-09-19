// r4d_quant_act_i8.hip - r4d_quant_act_i8: per-row symmetric int8 quantisation of a bf16 activation,
// written in the order r4d_gemm_w4a8_nt_m64's A fragment wants.
//
// One row per workgroup: absmax over K, then a second pass writing the bytes.  K is a few thousand
// and M is a speculative-decode batch, so the whole thing is well under a microsecond and exists
// only so the GEMM does not have to pay for a five-kernel torch expression.
//
// THE BYTE ORDER.  Fragment element e of lane l is A[row][16s + 8*(e>>2) + 4*(l>>4) + (e&3)], so a
// lane's eight bytes are two dwords eight bytes apart in a row-major activation.  Writing each k
// step's sixteen bytes as [half 0's eight][half 1's eight] instead makes them contiguous and the
// GEMM reads a fragment with one global_load_b64.  Rows are untouched, so every lane of a wave
// stays inside the same sixteen rows it would have read anyway -- laying A out as two separate
// PLANES is the version that loses, see the note in the GEMM.
#include "r4d.h"
#include <hip/hip_bf16.h>
#include <cstdint>
#include <stdexcept>

#define R4D_QAI8_THREADS 256

__global__ __launch_bounds__(R4D_QAI8_THREADS) void r4d_quant_act_i8_kernel(
    const __hip_bfloat16* __restrict__ A, signed char* __restrict__ Q, float* __restrict__ S,
    int K) {
  const int row = blockIdx.x;
  const __hip_bfloat16* a = A + (size_t)row * K;
  signed char* q = Q + (size_t)row * K;

  float m = 0.f;
  for (int i = threadIdx.x; i < K; i += R4D_QAI8_THREADS) {
    const float v = __bfloat162float(a[i]);
    m = fmaxf(m, fabsf(v));
  }
  __shared__ float red[R4D_QAI8_THREADS / 32];
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, 32));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
  __syncthreads();
  m = red[0];
#pragma unroll
  for (int i = 1; i < R4D_QAI8_THREADS / 32; i++) m = fmaxf(m, red[i]);
  const float scale = fmaxf(m, 1e-8f) / 127.f;
  const float inv = 1.f / scale;
  if (threadIdx.x == 0) S[row] = scale;

  // Byte j of a k step comes from lane-half j>>3 and element j&7, i.e. source 8*((j&7)>>2) +
  // 4*(j>>3) + (j&3) inside the step.
  for (int i = threadIdx.x; i < K; i += R4D_QAI8_THREADS) {
    const int s = i >> 4, j = i & 15;
    const int src = (s << 4) + 8 * ((j & 7) >> 2) + 4 * (j >> 3) + (j & 3);
    const float v = __bfloat162float(a[src]) * inv;
    const int t = __float2int_rn(v);
    q[i] = (signed char)(t < -128 ? -128 : (t > 127 ? 127 : t));
  }
}

void r4d_quant_act_i8(int64_t a, int64_t q, int64_t s, int M, int K, int64_t stream) {
  if (K % 16) throw std::runtime_error("r4d quant_act_i8: K must be a multiple of 16");
  if (M < 1) throw std::runtime_error("r4d quant_act_i8: M must be positive");
  hipLaunchKernelGGL(r4d_quant_act_i8_kernel, dim3(M), dim3(R4D_QAI8_THREADS), 0,
                     (hipStream_t)stream, (const __hip_bfloat16*)a, (signed char*)q, (float*)s, K);
}
