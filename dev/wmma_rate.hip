// Raw issue rate of every gfx1201 WMMA the drafter could use. Independent accumulator chains so
// the loop measures issue throughput, not latency; the trip count is a runtime arg so the compiler
// cannot fold it.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <vector>
typedef _Float16 v8h  __attribute__((ext_vector_type(8)));
typedef float    v8f  __attribute__((ext_vector_type(8)));
typedef int      v8i  __attribute__((ext_vector_type(8)));
typedef int      v2i  __attribute__((ext_vector_type(2)));
#define CH 8
#define UN 8

template <int OP>
__global__ __launch_bounds__(256) void rate(int iters, float* sink) {
  v8f  af[CH]; v8i ai[CH];
  v8h  a = (v8h)(_Float16)1.0f, b = (v8h)(_Float16)1.0f;
  v2i  q; q[0] = 0x11111111; q[1] = 0x22222222;
  int  n1 = 0x33333333;
#pragma unroll
  for (int i = 0; i < CH; i++) { af[i] = (v8f)(0.f); ai[i] = (v8i)(0); }
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int u = 0; u < UN; u++)
#pragma unroll
      for (int i = 0; i < CH; i++) {
        if      constexpr (OP == 0) af[i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, af[i]);
        else if constexpr (OP == 1) af[i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(q, q, af[i]);
        else if constexpr (OP == 2) ai[i] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, q, false, q, ai[i], false);
        else if constexpr (OP == 3) ai[i] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(false, q, false, q, ai[i], false);
        else if constexpr (OP == 4) ai[i] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(false, n1, false, n1, ai[i], false);
      }
  }
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < CH; i++) s += (OP == 0 || OP == 1) ? af[i][0] : (float)ai[i][0];
  sink[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

struct Case { const char* name; int op; double k; };

int main() {
  const Case cs[] = {
      {"v_wmma_f32_16x16x16_f16",     0, 16},
      {"v_wmma_f32_16x16x16_fp8_fp8", 1, 16},
      {"v_wmma_i32_16x16x16_iu8",     2, 16},
      {"v_wmma_i32_16x16x32_iu4",     3, 32},
      {"v_wmma_i32_16x16x16_iu4",     4, 16},
  };
  hipDeviceProp_t pr; hipGetDeviceProperties(&pr, 0);
  const int blocks = pr.multiProcessorCount * 8, iters = 2000;
  float* sink; hipMalloc(&sink, (size_t)blocks * 256 * 4);
  printf("gfx%s  %d CUs  %.2f GHz   %d blocks x 256 thr, %d iters x %d x %d WMMA\n",
         pr.gcnArchName, pr.multiProcessorCount, pr.clockRate / 1e6, blocks, iters, UN, CH);
  for (auto& c : cs) {
    auto run = [&](int n) {
      for (int i = 0; i < n; i++) switch (c.op) {
        case 0: hipLaunchKernelGGL((rate<0>), blocks, 256, 0, 0, iters, sink); break;
        case 1: hipLaunchKernelGGL((rate<1>), blocks, 256, 0, 0, iters, sink); break;
        case 2: hipLaunchKernelGGL((rate<2>), blocks, 256, 0, 0, iters, sink); break;
        case 3: hipLaunchKernelGGL((rate<3>), blocks, 256, 0, 0, iters, sink); break;
        case 4: hipLaunchKernelGGL((rate<4>), blocks, 256, 0, 0, iters, sink); break;
      }
      hipDeviceSynchronize();
    };
    run(3);
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    hipEventRecord(e0); run(20); hipEventRecord(e1); hipEventSynchronize(e1);
    float ms; hipEventElapsedTime(&ms, e0, e1);
    const double waves = (double)blocks * 256 / 32;
    const double instr = waves * iters * UN * CH * 20;
    const double flop  = instr * 2 * 16 * 16 * c.k;
    const double secs  = ms / 1e3;
    const double cyc   = secs * (pr.clockRate * 1e3) * pr.multiProcessorCount * 2 / (instr);
    printf("  %-30s %8.2f ms   %7.1f TOP/s   %5.2f cyc/instr/SIMD (2 SIMD/CU)\n",
           c.name, ms, flop / secs / 1e12, cyc);
  }
  return 0;
}
