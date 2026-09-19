// wmma_probe.hip -- probes which gfx12 (RDNA4/gfx1201) WMMA clang builtins compile, and for
// the ones that do, runs a register-resident dependent-free WMMA loop across all CUs to measure
// peak TFLOPS/TOPS per type.
//
// This is deliberately compiled ONCE PER VARIANT (see build.ps1, which passes -DWMMA_KIND=<n>)
// so that a compile failure on one variant does not block the others -- exactly the "probe which
// compile" methodology the task brief asks for. WMMA_KIND selects both the builtin called and the
// operand/result vector types, since gfx12 register packing differs per element type (see the
// per-variant comment below for the element-count arithmetic that determines each packing).
//
// Candidates are the gfx12-suffixed ("_w32_gfx12") wave32 builtins listed in this clang's own
// clang/Basic/BuiltinsAMDGPU.inc (verified present at
// C:\opt\rocm\lib\llvm\include\clang\Basic\BuiltinsAMDGPU.inc via grep before writing this file) --
// gfx1201 is wave32-native RDNA4, so the "_w32_gfx12" builtins (not the plain "_w32"/"_w64" ones,
// which are the older gfx11/RDNA3 encodings) are the intended targets. Signatures below are a
// best-effort reconstruction from the general WMMA register-packing convention (A/B packed to
// M*K/32 elements per lane, C/D to M*N/32 elements per lane); if a signature is wrong for this
// specific clang, the compile fails and the exact diagnostic is captured by build.ps1 -- itself
// useful ISA documentation per the task brief.
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

typedef float v8f __attribute__((ext_vector_type(8)));
typedef _Float16 v8h __attribute__((ext_vector_type(8)));
typedef short v8s __attribute__((ext_vector_type(8)));   // bf16 packed as 8 x bfloat16_t reinterpreted as short
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));

#ifndef WMMA_KIND
#define WMMA_KIND 0
#endif

// Independent accumulator chains: consecutive WMMA issues cycle through NCHAINS separate
// accumulators so each instruction's inputs do not depend on the immediately preceding
// instruction's result -- this is what makes the loop throughput- rather than latency-bound.
// (An earlier version of this file used a single accumulator chain, i.e. read-after-write on every
// instruction; that measured WMMA pipeline *latency*, not issue throughput, and produced near-
// identical numbers for every dtype -- a tell documented in docs/research/r9700-microbench.md.)
#define NCHAINS 8
#define DEPTH 8
#define UNROLL (NCHAINS * DEPTH)

#if WMMA_KIND == 0
// f32_16x16x16_f16: A,B = 8 x f16/lane (v8h), C,D = 8 x f32/lane (v8f)
#define KIND_NAME "f32_16x16x16_f16"
#define FLOPS_PER_WMMA (2ull * 16 * 16 * 16)
typedef v8h A_T; typedef v8h B_T; typedef v8f OUT_T;
#define ELEM_T(x) ((float)(x))
__global__ void wmma_kernel(const v8h* __restrict__ A, const v8h* __restrict__ B, v8f* __restrict__ Dout) {
    v8h a = A[threadIdx.x % 32];
    v8h b = B[threadIdx.x % 32];
    v8f d[NCHAINS];
#pragma unroll
    for (int c = 0; c < NCHAINS; ++c) d[c] = {ELEM_T(c + 1), 0, 0, 0, 0, 0, 0, 0}; // distinct per chain: prevents LLVM CSE from merging the "independent" chains into one (all chains would otherwise start from literally the same accumulator value with the same a/b operands, since the operand buffers are also a uniform memset pattern)
#pragma unroll
    for (int i = 0; i < DEPTH; ++i) {
#pragma unroll
        for (int c = 0; c < NCHAINS; ++c) {
            d[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, d[c]);
        }
    }
    // Always-executed write (NOT behind an `if (threadIdx.x == some-unreachable-value)` guard --
    // an earlier version of this file used such a guard, which is provably always-false at
    // compile time since threadIdx.x < blockDim.x <= 256, so LLVM proved the whole computation
    // dead and emitted an empty kernel body (`s_endpgm`, confirmed via -S) -- every number from
    // that version was launch-overhead noise, not WMMA throughput; see
    // docs/research/r9700-microbench.md for how this was caught (disassembly inspection).
    v8f acc = d[0];
#pragma unroll
    for (int c = 1; c < NCHAINS; ++c) { acc += d[c]; }
    Dout[blockIdx.x * blockDim.x + threadIdx.x] = acc; // never taken, keeps every chain live
}

#elif WMMA_KIND == 1
// f32_16x16x16_bf16: A,B = 8 x bf16/lane packed as v8s (reinterpret), C,D = v8f
#define KIND_NAME "f32_16x16x16_bf16"
#define FLOPS_PER_WMMA (2ull * 16 * 16 * 16)
typedef v8s A_T; typedef v8s B_T; typedef v8f OUT_T;
#define ELEM_T(x) ((float)(x))
__global__ void wmma_kernel(const v8s* __restrict__ A, const v8s* __restrict__ B, v8f* __restrict__ Dout) {
    v8s a = A[threadIdx.x % 32];
    v8s b = B[threadIdx.x % 32];
    v8f d[NCHAINS];
#pragma unroll
    for (int c = 0; c < NCHAINS; ++c) d[c] = {ELEM_T(c + 1), 0, 0, 0, 0, 0, 0, 0}; // distinct per chain: prevents LLVM CSE from merging the "independent" chains into one (all chains would otherwise start from literally the same accumulator value with the same a/b operands, since the operand buffers are also a uniform memset pattern)
#pragma unroll
    for (int i = 0; i < DEPTH; ++i) {
#pragma unroll
        for (int c = 0; c < NCHAINS; ++c) {
            d[c] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, d[c]);
        }
    }
    // Always-executed write (NOT behind an `if (threadIdx.x == some-unreachable-value)` guard --
    // an earlier version of this file used such a guard, which is provably always-false at
    // compile time since threadIdx.x < blockDim.x <= 256, so LLVM proved the whole computation
    // dead and emitted an empty kernel body (`s_endpgm`, confirmed via -S) -- every number from
    // that version was launch-overhead noise, not WMMA throughput; see
    // docs/research/r9700-microbench.md for how this was caught (disassembly inspection).
    v8f acc = d[0];
#pragma unroll
    for (int c = 1; c < NCHAINS; ++c) { acc += d[c]; }
    Dout[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

#elif WMMA_KIND == 2
// f32_16x16x16_fp8_fp8: A,B = 8 bytes/lane packed as v2i, C,D = v8f
#define KIND_NAME "f32_16x16x16_fp8_fp8"
#define FLOPS_PER_WMMA (2ull * 16 * 16 * 16)
typedef v2i A_T; typedef v2i B_T; typedef v8f OUT_T;
#define ELEM_T(x) ((float)(x))
__global__ void wmma_kernel(const v2i* __restrict__ A, const v2i* __restrict__ B, v8f* __restrict__ Dout) {
    v2i a = A[threadIdx.x % 32];
    v2i b = B[threadIdx.x % 32];
    v8f d[NCHAINS];
#pragma unroll
    for (int c = 0; c < NCHAINS; ++c) d[c] = {ELEM_T(c + 1), 0, 0, 0, 0, 0, 0, 0}; // distinct per chain: prevents LLVM CSE from merging the "independent" chains into one (all chains would otherwise start from literally the same accumulator value with the same a/b operands, since the operand buffers are also a uniform memset pattern)
#pragma unroll
    for (int i = 0; i < DEPTH; ++i) {
#pragma unroll
        for (int c = 0; c < NCHAINS; ++c) {
            d[c] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, d[c]);
        }
    }
    // Always-executed write (NOT behind an `if (threadIdx.x == some-unreachable-value)` guard --
    // an earlier version of this file used such a guard, which is provably always-false at
    // compile time since threadIdx.x < blockDim.x <= 256, so LLVM proved the whole computation
    // dead and emitted an empty kernel body (`s_endpgm`, confirmed via -S) -- every number from
    // that version was launch-overhead noise, not WMMA throughput; see
    // docs/research/r9700-microbench.md for how this was caught (disassembly inspection).
    v8f acc = d[0];
#pragma unroll
    for (int c = 1; c < NCHAINS; ++c) { acc += d[c]; }
    Dout[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

#elif WMMA_KIND == 3
// f32_16x16x16_bf8_bf8: same packing as fp8_fp8
#define KIND_NAME "f32_16x16x16_bf8_bf8"
#define FLOPS_PER_WMMA (2ull * 16 * 16 * 16)
typedef v2i A_T; typedef v2i B_T; typedef v8f OUT_T;
#define ELEM_T(x) ((float)(x))
__global__ void wmma_kernel(const v2i* __restrict__ A, const v2i* __restrict__ B, v8f* __restrict__ Dout) {
    v2i a = A[threadIdx.x % 32];
    v2i b = B[threadIdx.x % 32];
    v8f d[NCHAINS];
#pragma unroll
    for (int c = 0; c < NCHAINS; ++c) d[c] = {ELEM_T(c + 1), 0, 0, 0, 0, 0, 0, 0}; // distinct per chain: prevents LLVM CSE from merging the "independent" chains into one (all chains would otherwise start from literally the same accumulator value with the same a/b operands, since the operand buffers are also a uniform memset pattern)
#pragma unroll
    for (int i = 0; i < DEPTH; ++i) {
#pragma unroll
        for (int c = 0; c < NCHAINS; ++c) {
            d[c] = __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(a, b, d[c]);
        }
    }
    // Always-executed write (NOT behind an `if (threadIdx.x == some-unreachable-value)` guard --
    // an earlier version of this file used such a guard, which is provably always-false at
    // compile time since threadIdx.x < blockDim.x <= 256, so LLVM proved the whole computation
    // dead and emitted an empty kernel body (`s_endpgm`, confirmed via -S) -- every number from
    // that version was launch-overhead noise, not WMMA throughput; see
    // docs/research/r9700-microbench.md for how this was caught (disassembly inspection).
    v8f acc = d[0];
#pragma unroll
    for (int c = 1; c < NCHAINS; ++c) { acc += d[c]; }
    Dout[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

#elif WMMA_KIND == 4
// i32_16x16x16_iu8: A,B packed 8 bytes/lane (v2i) + a sign bool each, C,D = v8i, trailing clamp bool
#define KIND_NAME "i32_16x16x16_iu8"
#define FLOPS_PER_WMMA (2ull * 16 * 16 * 16) // "ops" (MACs*2), reported as TOPS not TFLOPS
typedef v2i A_T; typedef v2i B_T; typedef v8i OUT_T;
#define ELEM_T(x) (x)
__global__ void wmma_kernel(const v2i* __restrict__ A, const v2i* __restrict__ B, v8i* __restrict__ Dout) {
    v2i a = A[threadIdx.x % 32];
    v2i b = B[threadIdx.x % 32];
    v8i d[NCHAINS];
#pragma unroll
    for (int c = 0; c < NCHAINS; ++c) d[c] = {ELEM_T(c + 1), 0, 0, 0, 0, 0, 0, 0}; // distinct per chain: prevents LLVM CSE from merging the "independent" chains into one (all chains would otherwise start from literally the same accumulator value with the same a/b operands, since the operand buffers are also a uniform memset pattern)
#pragma unroll
    for (int i = 0; i < DEPTH; ++i) {
#pragma unroll
        for (int c = 0; c < NCHAINS; ++c) {
            d[c] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, d[c], false);
        }
    }
    // Always-executed write -- see the f32 variants' identical comment above for why this must
    // not be behind a compile-time-provably-false guard.
    v8i acc = d[0];
#pragma unroll
    for (int c = 1; c < NCHAINS; ++c) { acc += d[c]; }
    Dout[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

#elif WMMA_KIND == 5
// i32_16x16x16_iu4: A,B packed 8 x 4-bit/lane = 4 bytes = single int, C,D = v8i, trailing clamp bool
#define KIND_NAME "i32_16x16x16_iu4"
#define FLOPS_PER_WMMA (2ull * 16 * 16 * 16)
typedef int A_T; typedef int B_T; typedef v8i OUT_T;
#define ELEM_T(x) (x)
__global__ void wmma_kernel(const int* __restrict__ A, const int* __restrict__ B, v8i* __restrict__ Dout) {
    int a = A[threadIdx.x % 32];
    int b = B[threadIdx.x % 32];
    v8i d[NCHAINS];
#pragma unroll
    for (int c = 0; c < NCHAINS; ++c) d[c] = {ELEM_T(c + 1), 0, 0, 0, 0, 0, 0, 0}; // distinct per chain: prevents LLVM CSE from merging the "independent" chains into one (all chains would otherwise start from literally the same accumulator value with the same a/b operands, since the operand buffers are also a uniform memset pattern)
#pragma unroll
    for (int i = 0; i < DEPTH; ++i) {
#pragma unroll
        for (int c = 0; c < NCHAINS; ++c) {
            d[c] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(true, a, true, b, d[c], false);
        }
    }
    // Always-executed write -- see the f32 variants' identical comment above for why this must
    // not be behind a compile-time-provably-false guard.
    v8i acc = d[0];
#pragma unroll
    for (int c = 1; c < NCHAINS; ++c) { acc += d[c]; }
    Dout[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

#elif WMMA_KIND == 6
// i32_16x16x32_iu4 (double-pumped K=32): A,B packed 16 x 4-bit/lane = 8 bytes = v2i, C,D = v8i
#define KIND_NAME "i32_16x16x32_iu4"
#define FLOPS_PER_WMMA (2ull * 16 * 16 * 32)
typedef v2i A_T; typedef v2i B_T; typedef v8i OUT_T;
#define ELEM_T(x) (x)
__global__ void wmma_kernel(const v2i* __restrict__ A, const v2i* __restrict__ B, v8i* __restrict__ Dout) {
    v2i a = A[threadIdx.x % 32];
    v2i b = B[threadIdx.x % 32];
    v8i d[NCHAINS];
#pragma unroll
    for (int c = 0; c < NCHAINS; ++c) d[c] = {ELEM_T(c + 1), 0, 0, 0, 0, 0, 0, 0}; // distinct per chain: prevents LLVM CSE from merging the "independent" chains into one (all chains would otherwise start from literally the same accumulator value with the same a/b operands, since the operand buffers are also a uniform memset pattern)
#pragma unroll
    for (int i = 0; i < DEPTH; ++i) {
#pragma unroll
        for (int c = 0; c < NCHAINS; ++c) {
            d[c] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, a, true, b, d[c], false);
        }
    }
    // Always-executed write -- see the f32 variants' identical comment above for why this must
    // not be behind a compile-time-provably-false guard.
    v8i acc = d[0];
#pragma unroll
    for (int c = 1; c < NCHAINS; ++c) { acc += d[c]; }
    Dout[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

#endif

int main(int argc, char** argv) {
    int iters = (argc > 1) ? atoi(argv[1]) : 200;
    int block = 256;   // 8 warps/wavefronts of 32
    int grid = 64 * 16; // heavily oversubscribe 64 CUs

    // Allocate small dummy operand buffers (32 lanes worth); contents don't matter for a
    // throughput probe (values are fixed/never read back).
    // Dout is now written by EVERY thread (see the kernel's own comment on why the write can't be
    // behind a compile-time-provably-unreachable guard), so it needs grid*block entries, not just
    // grid.
    void* dA; void* dB; void* dOut;
    HIP_CHECK(hipMalloc(&dA, 32 * 16));
    HIP_CHECK(hipMalloc(&dB, 32 * 16));
    HIP_CHECK(hipMalloc(&dOut, (size_t)grid * block * 32));
    HIP_CHECK(hipMemset(dA, 0x11, 32 * 16));
    HIP_CHECK(hipMemset(dB, 0x11, 32 * 16));
    HIP_CHECK(hipMemset(dOut, 0, (size_t)grid * block * 32));

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    A_T* pA = reinterpret_cast<A_T*>(dA);
    B_T* pB = reinterpret_cast<B_T*>(dB);
    OUT_T* pOut = reinterpret_cast<OUT_T*>(dOut);

    for (int w = 0; w < 3; ++w) {
        wmma_kernel<<<dim3(grid), dim3(block)>>>(pA, pB, pOut);
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<double> results;
    for (int it = 0; it < 5; ++it) {
        HIP_CHECK(hipEventRecord(start, 0));
        for (int r = 0; r < iters; ++r) {
            wmma_kernel<<<dim3(grid), dim3(block)>>>(pA, pB, pOut);
        }
        HIP_CHECK(hipEventRecord(stop, 0));
        HIP_CHECK(hipEventSynchronize(stop));
        float ms = 0.f;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        unsigned long long total_wmma_ops = (unsigned long long)iters * grid * (block / 32) * UNROLL;
        double total_flops = (double)total_wmma_ops * (double)FLOPS_PER_WMMA;
        double tflops = total_flops / (ms / 1000.0) / 1e12;
        results.push_back(tflops);
    }
    std::sort(results.begin(), results.end());
    double med = results[results.size() / 2];
    printf("variant=%s median_TFLOPS_or_TOPS=%.3f raw=[", KIND_NAME, med);
    for (double v : results) printf("%.3f ", v);
    printf("]\n");

    return 0;
}
