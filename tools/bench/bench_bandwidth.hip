// bench_bandwidth.hip -- streaming VRAM bandwidth microbenchmark for the R9700 (gfx1201).
//
// Two kernels:
//   read_kernel : every thread streams int4 (16B) vector loads across a buffer and folds them
//                 into a per-thread checksum that is atomically added into a single output int4,
//                 so the compiler cannot dead-code-eliminate the loads. Measures pure read
//                 bandwidth.
//   copy_kernel : every thread reads int4 from src and writes int4 to dst (read+write bandwidth,
//                 the "copy" figure vendors usually quote).
//
// Usage: bench_bandwidth.exe [iters]
// Prints CSV rows: size_bytes,mode,GB/s  (one row per (size,mode) pair, iters repeats internally
// and prints the median of the *internal* iters -- the external driver (run_bench.ps1) additionally
// invokes this whole binary 3x and takes the median across process runs per the task brief).
//
// Build: see build.ps1 in this directory.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <chrono>

#define HIP_CHECK(expr) do { \
    hipError_t _e = (expr); \
    if (_e != hipSuccess) { \
      fprintf(stderr, "HIP error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__, hipGetErrorString(_e)); \
      exit(1); \
    } \
  } while (0)

__global__ void read_kernel(const int4* __restrict__ src, unsigned long long n_vec, int4* __restrict__ out) {
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    int4 acc = make_int4(0, 0, 0, 0);
    for (unsigned long long idx = i; idx < n_vec; idx += stride) {
        int4 v = src[idx];
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    // Reduce into a tiny output buffer so the loads survive the optimizer, without a real
    // cross-thread reduction dominating the timing (only 1 CAS-free atomic per thread).
    atomicXor(&out[0].x, acc.x);
    atomicXor(&out[0].y, acc.y);
    atomicXor(&out[0].z, acc.z);
    atomicXor(&out[0].w, acc.w);
}

__global__ void copy_kernel(const int4* __restrict__ src, int4* __restrict__ dst, unsigned long long n_vec) {
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned long long idx = i; idx < n_vec; idx += stride) {
        dst[idx] = src[idx];
    }
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

int main(int argc, char** argv) {
    int iters = (argc > 1) ? atoi(argv[1]) : 10;
    int warmup = 3;

    // gfx1201 has 64 CUs; occupy all of them generously.
    int block = 256;
    int grid = 64 * 8; // 512 blocks, plenty to saturate 64 CUs with this occupancy

    std::vector<unsigned long long> sizes = {
        16ull * 1024 * 1024,        // 16 MB
        48ull * 1024 * 1024,        // 48 MB (Infinity Cache sized)
        256ull * 1024 * 1024,       // 256 MB
        2ull * 1024 * 1024 * 1024,  // 2 GB
        8ull * 1024 * 1024 * 1024,  // 8 GB
    };

    printf("size_bytes,mode,GB/s\n");

    for (unsigned long long bytes : sizes) {
        unsigned long long n_vec = bytes / 16;
        bytes = n_vec * 16; // round to whole int4s

        void* d_src = nullptr;
        void* d_out = nullptr;
        HIP_CHECK(hipMalloc(&d_src, bytes));
        HIP_CHECK(hipMemset(d_src, 0x5a, bytes));
        HIP_CHECK(hipMalloc(&d_out, sizeof(int4)));
        HIP_CHECK(hipMemset(d_out, 0, sizeof(int4)));

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        // ---- read-only bandwidth ----
        for (int w = 0; w < warmup; ++w) {
            hipLaunchKernelGGL(read_kernel, dim3(grid), dim3(block), 0, 0,
                                (const int4*)d_src, n_vec, (int4*)d_out);
        }
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<double> gbps_read;
        for (int it = 0; it < iters; ++it) {
            HIP_CHECK(hipEventRecord(start, 0));
            hipLaunchKernelGGL(read_kernel, dim3(grid), dim3(block), 0, 0,
                                (const int4*)d_src, n_vec, (int4*)d_out);
            HIP_CHECK(hipEventRecord(stop, 0));
            HIP_CHECK(hipEventSynchronize(stop));
            float ms = 0.f;
            HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
            double gbps = (double)bytes / (ms / 1000.0) / 1e9;
            gbps_read.push_back(gbps);
        }
        printf("%llu,read,%.2f\n", bytes, median(gbps_read));
        fflush(stdout);

        // ---- copy (read+write) bandwidth: needs a same-size dst ----
        void* d_dst = nullptr;
        HIP_CHECK(hipMalloc(&d_dst, bytes));
        for (int w = 0; w < warmup; ++w) {
            hipLaunchKernelGGL(copy_kernel, dim3(grid), dim3(block), 0, 0,
                                (const int4*)d_src, (int4*)d_dst, n_vec);
        }
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<double> gbps_copy;
        for (int it = 0; it < iters; ++it) {
            HIP_CHECK(hipEventRecord(start, 0));
            hipLaunchKernelGGL(copy_kernel, dim3(grid), dim3(block), 0, 0,
                                (const int4*)d_src, (int4*)d_dst, n_vec);
            HIP_CHECK(hipEventRecord(stop, 0));
            HIP_CHECK(hipEventSynchronize(stop));
            float ms = 0.f;
            HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
            // copy touches 2x bytes (1 read + 1 write)
            double gbps = (2.0 * (double)bytes) / (ms / 1000.0) / 1e9;
            gbps_copy.push_back(gbps);
        }
        printf("%llu,copy,%.2f\n", bytes, median(gbps_copy));
        fflush(stdout);

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        HIP_CHECK(hipFree(d_src));
        HIP_CHECK(hipFree(d_dst));
        HIP_CHECK(hipFree(d_out));
    }

    return 0;
}
