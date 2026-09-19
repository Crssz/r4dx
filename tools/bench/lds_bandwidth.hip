// lds_bandwidth.hip -- LDS (shared memory) bandwidth microbenchmark: a ds_read_b128-style
// (128-bit, i.e. int4) load loop per CU, one workgroup per CU, timed with hipEvents.
//
// Usage: lds_bandwidth.exe [iters]
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define HIP_CHECK(expr) do { \
    hipError_t _e = (expr); \
    if (_e != hipSuccess) { \
      fprintf(stderr, "HIP error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__, hipGetErrorString(_e)); \
      exit(1); \
    } \
  } while (0)

#define LDS_INT4_COUNT 1024  // 1024 * 16B = 16 KiB tile, fits comfortably in gfx1201's LDS/CU
#define REPS 4096             // loads per thread per kernel launch

__global__ void __launch_bounds__(256) lds_read_kernel(int4* __restrict__ out) {
    __shared__ int4 tile[LDS_INT4_COUNT];
    int tid = threadIdx.x;
    // Initialize once per block (not timed contribution matters relative to REPS>>tile size).
    for (int i = tid; i < LDS_INT4_COUNT; i += blockDim.x) {
        tile[i] = make_int4(i, i, i, i);
    }
    __syncthreads();

    int4 acc = make_int4(0, 0, 0, 0);
    int idx = tid % LDS_INT4_COUNT;
#pragma unroll 8
    for (int r = 0; r < REPS; ++r) {
        int4 v = tile[idx];
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
        idx = (idx + 1) % LDS_INT4_COUNT;
    }
    // Always-executed write: an earlier version guarded this with
    // `if (tid == 0xffffffff)`, which is provably always-false at compile time (tid < blockDim.x
    // <= 256), so LLVM proved the whole kernel body dead and emitted an empty kernel -- confirmed
    // via -S disassembly the same way the wmma_probe.hip bug was caught (see that file's comment
    // and docs/research/r9700-microbench.md). Every thread now writes its own slot.
    out[blockIdx.x * blockDim.x + tid] = acc;
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

int main(int argc, char** argv) {
    int iters = (argc > 1) ? atoi(argv[1]) : 20;
    int block = 256;
    int grid = 64; // one workgroup per CU (gfx1201 has 64 CUs)

    void* d_out;
    HIP_CHECK(hipMalloc(&d_out, (size_t)grid * block * sizeof(int4)));
    HIP_CHECK(hipMemset(d_out, 0, (size_t)grid * block * sizeof(int4)));

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    for (int w = 0; w < 3; ++w) {
        hipLaunchKernelGGL(lds_read_kernel, dim3(grid), dim3(block), 0, 0, (int4*)d_out);
    }
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<double> gbps_trials;
    for (int it = 0; it < iters; ++it) {
        HIP_CHECK(hipEventRecord(start, 0));
        hipLaunchKernelGGL(lds_read_kernel, dim3(grid), dim3(block), 0, 0, (int4*)d_out);
        HIP_CHECK(hipEventRecord(stop, 0));
        HIP_CHECK(hipEventSynchronize(stop));
        float ms = 0.f;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        unsigned long long bytes = (unsigned long long)grid * block * REPS * 16ull;
        double gbps = (double)bytes / (ms / 1000.0) / 1e9;
        gbps_trials.push_back(gbps);
    }
    printf("lds_read_b128: GBps_median=%.2f raw=[", median(gbps_trials));
    for (double v : gbps_trials) printf("%.2f ", v);
    printf("]\n");

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(d_out));
    return 0;
}
