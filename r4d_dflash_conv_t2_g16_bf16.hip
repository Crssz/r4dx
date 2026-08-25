// DFlash2 grouped dynamic depthwise convolution, fused.
//
// The device math lives in r4d_dflash_conv_body.h, deliberately free of HIP headers so the same
// source compiles standalone to gfx1201 assembly with a stock clang for ISA review.
#include <hip/hip_runtime.h>
#include "r4d_dflash_conv_body.h"

#define DFC_THREADS 128
#define DFC_VEC     8      // bf16 lanes per thread: one dwordx4 in and one out

template <int TAPS, int GROUP>
__global__ __launch_bounds__(DFC_THREADS) void r4d_dflash_conv_kernel(
    const r4d_u16* __restrict__ x, const r4d_u16* __restrict__ delta,
    const r4d_u16* __restrict__ base, r4d_u16* __restrict__ out,
    int H, int dpitch, int NG, int vpr, int blockmask)
{
  const int hv = blockIdx.x * DFC_THREADS + threadIdx.x;
  r4d_dflash_conv_body<TAPS, GROUP>(x, delta, base, out, H, dpitch, NG, vpr,
                                    blockmask, blockIdx.y, hv);
}

// x     [T, H]        bf16, row pitch H
// delta [T, taps, NG] bf16, row pitch dpitch -- this is a SLICE of the [T, 2, taps, NG] kernel
//                     projection, so dpitch is 2*taps*NG, not taps*NG. Passing the packed pitch
//                     silently reads the other side's coefficients; it is not a shape error.
// base  [taps, H]     bf16, contiguous
// out   [T, H]        bf16. Must not alias x -- tap 1 of row t reads row t-1 of the INPUT.
extern "C" int r4d_dflash_conv_t2_g16_bf16(
    const void* x, const void* delta, const void* base, void* out,
    int T, int H, int dpitch, int NG, int taps, int group, int block_size, void* stream)
{
  if (taps != 2 || group != 16) return -1;
  if (H % (group * DFC_VEC) || NG != H / group) return -1;
  if (block_size < 1 || (block_size & (block_size - 1))) return -1;  // body masks: pow2 only
  if (T <= 0) return 0;

  const int vpr = H / DFC_VEC;
  dim3 grid((vpr + DFC_THREADS - 1) / DFC_THREADS, T), blk(DFC_THREADS);
  hipLaunchKernelGGL((r4d_dflash_conv_kernel<2, 16>), grid, blk, 0, (hipStream_t)stream,
                     (const r4d_u16*)x, (const r4d_u16*)delta, (const r4d_u16*)base,
                     (r4d_u16*)out, H, dpitch, NG, vpr, block_size - 1);
  return (int)hipGetLastError();
}
