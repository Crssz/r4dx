// The device side of the shader-clock probe and the GPU timer stamp (docs/trellis-kernel.md 4.6
// "Clock", 6 "Benches"), shared by src/model/kernels/model_kernels.hip (the in-model probe,
// r4dx's R4DX_CLOCK_PROBE / R4DX_PROFILE_LINEARS, src/model/debug_probe.h) and
// tests/kernels/tool_trellis_gemm_bench_kernels.hip (the replay's probe and timer), so the clock
// the model reports and the clock the replay reports come from one piece of device code. Each
// includer wraps these in its own __global__ kernels.
//
// clock64() is s_getreg SHADER_CYCLES (the shader clock, per wave) and wall_clock64() is
// s_sendmsg_rtn GET_REALTIME (a constant ~100 MHz), so d(clock64) / d(wall_clock64) * wall rate is
// the shader clock over that interval. The probe's loop is a dependent v_add chain, a fixed number
// of shader cycles per add whatever the clock.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

// out[0] = wall_clock64 when the calling thread runs this. Launched as a one-thread kernel, it is a
// timer stamp: kernels of one stream run in order, so two stamps bracket everything between them.
__device__ __forceinline__ void r4dx_wall_stamp_device(unsigned long long* out) {
  out[0] = wall_clock64();
}

// out[0..3] = clock64, wall_clock64 before and after a dependent chain of 8 * iters v_add; out[4] =
// the chain's value (keeps it alive). Launched as one wave of 32 threads.
__device__ __forceinline__ void r4dx_clock_probe_device(unsigned long long* out, int iters) {
  const unsigned long long c0 = clock64(), w0 = wall_clock64();
  uint32_t x = threadIdx.x;
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int j = 0; j < 8; ++j) asm volatile("v_add_nc_u32 %0, %0, 1" : "+v"(x));
  }
  const unsigned long long c1 = clock64(), w1 = wall_clock64();
  if (threadIdx.x == 0) {
    out[0] = c0;
    out[1] = w0;
    out[2] = c1;
    out[3] = w1;
    out[4] = x;
  }
}
