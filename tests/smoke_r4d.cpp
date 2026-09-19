// Toolchain smoke test: links r4d_core, prints the attention/GDN kernel geometry and the
// registry size, then runs r4d_gemm_bf16_nt_m64 on device 1 (HIP_VISIBLE_DEVICES=1, set by
// tests/run_tests.ps1, so device index 0 here IS physical device 1) against a CPU fp32
// reference. Passes if the relative error is under 2e-2 -- the same threshold and M/N/K/WV/SK/MB
// libr4d's own build-win/check_gemm_bf16.py uses.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "r4d.h"

namespace {

#define HIP_CHECK(expr)                                                                  \
    do {                                                                                 \
        hipError_t _err = (expr);                                                        \
        if (_err != hipSuccess) {                                                        \
            std::fprintf(stderr, "HIP error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__, \
                         hipGetErrorString(_err));                                       \
            std::exit(1);                                                                \
        }                                                                                 \
    } while (0)

// bf16 <-> fp32. bf16 is the top 16 bits of an IEEE-754 fp32 (round-to-nearest-even).
uint16_t FloatToBf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    uint32_t rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(rounded >> 16);
}

float Bf16ToFloat(uint16_t h) {
    uint32_t bits = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

}  // namespace

int main() {
    // ---- registry + geometry ---------------------------------------------------------------
    int head_dim = 0, gqa = 0, block_size = 0, max_decode_rows = 0;
    r4d_attn_dims(&head_dim, &gqa, &block_size, &max_decode_rows);
    std::printf("r4d_attn_dims: head_dim=%d gqa=%d block_size=%d max_decode_rows=%d\n", head_dim,
                gqa, block_size, max_decode_rows);

    int head_k = 0, head_v = 0, chunk = 0;
    r4d_gdn_dims(&head_k, &head_v, &chunk);
    std::printf("r4d_gdn_dims: head_k=%d head_v=%d chunk=%d\n", head_k, head_v, chunk);

    int n_kernels = r4d_kernel_count();
    std::printf("r4d_kernel_count: %d\n", n_kernels);
    for (int i = 0; i < n_kernels; ++i) {
        const R4DKernelInfo* k = r4d_kernel_at(i);
        std::printf("  [%2d] %-40s family=%-6s op=%s\n", i, k->name, k->family, k->op);
    }

    // ---- device 1 (HIP_VISIBLE_DEVICES=1 -> local index 0) ---------------------------------
    HIP_CHECK(hipSetDevice(0));

    // ---- r4d_gemm_bf16_nt_m64: C[M,N] = A[M,K] @ W[N,K]^T ------------------------------------
    // Same shape/tuning libr4d's build-win/check_gemm_bf16.py uses: K % (SK*16) == 0 and
    // WV*SK*32 <= 1024.
    const int M = 8, K = 2048, N = 1024;
    const int WV = 4, SK = 4, MB = 1;

    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-0.2f, 0.2f);

    std::vector<float> Af(M * K), Wf(N * K);
    std::vector<uint16_t> Ah(M * K), Wh(N * K);
    for (auto& v : Af) v = dist(rng);
    for (auto& v : Wf) v = dist(rng);
    for (size_t i = 0; i < Af.size(); ++i) Ah[i] = FloatToBf16(Af[i]);
    for (size_t i = 0; i < Wf.size(); ++i) Wh[i] = FloatToBf16(Wf[i]);

    void *dA = nullptr, *dW = nullptr, *dC = nullptr;
    HIP_CHECK(hipMalloc(&dA, Ah.size() * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&dW, Wh.size() * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&dC, static_cast<size_t>(M) * N * sizeof(uint16_t)));
    HIP_CHECK(hipMemcpy(dA, Ah.data(), Ah.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dW, Wh.data(), Wh.size() * sizeof(uint16_t), hipMemcpyHostToDevice));

    r4d_gemm_bf16_nt_m64(reinterpret_cast<int64_t>(dA), reinterpret_cast<int64_t>(dW),
                          reinterpret_cast<int64_t>(dC), M, K, N, WV, SK, MB, /*stream=*/0);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<uint16_t> Ch(static_cast<size_t>(M) * N);
    HIP_CHECK(hipMemcpy(Ch.data(), dC, Ch.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dW));
    HIP_CHECK(hipFree(dC));

    // CPU fp32 reference: A (bf16-rounded) @ W (bf16-rounded)^T.
    double num = 0.0, den = 0.0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += Bf16ToFloat(Ah[m * K + k]) * Bf16ToFloat(Wh[n * K + k]);
            }
            float got = Bf16ToFloat(Ch[m * N + n]);
            double d = static_cast<double>(got) - static_cast<double>(acc);
            num += d * d;
            den += static_cast<double>(acc) * static_cast<double>(acc);
        }
    }
    double rel_err = std::sqrt(num) / std::sqrt(den);
    std::printf("gemm_bf16_nt_m64 M=%d N=%d K=%d WV=%d SK=%d MB=%d: rel_err=%.4e\n", M, N, K, WV,
                SK, MB, rel_err);

    if (!(rel_err < 2e-2)) {
        std::fprintf(stderr, "FAIL: rel_err %.4e >= 2e-2\n", rel_err);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
