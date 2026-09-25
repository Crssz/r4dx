// r4dx::kernels::hadamard -- DEVICE-ONLY helpers shared by the quant2 rotation kernels
// (docs/quant2.md sections 3-4): src/kernels/src/rotate_residual.hip (Q / Q^T on the residual
// stream, the gdn.out_proj in-place Hb), src/kernels/src/r4dx_kernels.hip (silu_mul's Hb variant)
// and src/model/attention/src/attn_kernels.hip (the output-gate multiply's Hb variant). One copy of
// the butterfly so the three consumers cannot drift apart. hipcc translation units only -- nothing
// in here is callable from the clang-cl host build, which is why it is not part of kernels.h.
//
// Transform convention (the quant2 rotation contract): natural (Sylvester) order, H[i][j] =
// (-1)^popcount(i & j), UNNORMALIZED -- the caller applies 1/sqrt(B) (or 1/32 for B = 1024) itself.
// H_B H_B = B * I, so the normalized transform is self-inverse.
#pragma once

#if !defined(__HIPCC__)
#error "r4dx/kernels/hadamard_device.h is device code: include it only from a hipcc (.hip) TU"
#endif

#include <hip/hip_runtime.h>

#include <cstdint>

namespace r4dx::kernels::hadamard {

// In-place unnormalized fast Walsh-Hadamard transform of `n` contiguous floats in LDS, as n / B
// independent blocks of B = 1 << log2_block (n must be a multiple of B; a butterfly never crosses a
// block because every pair (i, i + h) with h < B agrees on all bits >= log2_block).
//
// Every thread of the workgroup must call this: it holds a __syncthreads() after each of the
// log2_block stages, including the last, so on return `v` is complete and readable by every thread.
// The caller owns the barrier BEFORE the call (between filling `v` and the first stage).
//
// Deterministic and independent of blockDim / grid shape: each output is produced by the same
// fixed sequence of fp32 adds/subtracts whichever thread happens to execute a given butterfly, so a
// row's result never depends on how many rows or blocks were launched alongside it (the TP=1
// identity baseline and f7d4927's "verify rows sum in decode order" exactness rely on that).
__device__ __forceinline__ void FwhtLds(float* v, int n, int log2_block) {
  const int half = n >> 1;
  for (int lg = 0; lg < log2_block; ++lg) {
    const int h = 1 << lg;
    for (int p = static_cast<int>(threadIdx.x); p < half; p += static_cast<int>(blockDim.x)) {
      // Insert a 0 bit at position lg of p: the lower element of this stage's p-th pair.
      const int i = ((p >> lg) << (lg + 1)) | (p & (h - 1));
      const float a = v[i];
      const float b = v[i + h];
      v[i] = a + b;
      v[i + h] = a - b;
    }
    __syncthreads();
  }
}

// Host-side helpers for the entry points' precondition checks and launch shapes.
inline bool IsPow2(int64_t v) { return v > 0 && (v & (v - 1)) == 0; }
inline int Log2Pow2(int64_t v) {
  int lg = 0;
  while ((int64_t{1} << lg) < v) ++lg;
  return lg;
}
// One butterfly per thread per stage, clamped to [32, 256] (a wave32 minimum; 256 = the r4dx
// kernels' usual workgroup): B = 128 -> 64 threads, 256 -> 128, 512 and 1024 -> 256.
inline int ThreadsForBlock(int64_t block) {
  int64_t t = block / 2;
  if (t < 32) t = 32;
  if (t > 256) t = 256;
  return static_cast<int>(t);
}

}  // namespace r4dx::kernels::hadamard
