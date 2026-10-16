#include "r4dx/model/qwen4exp/hc_bf16_gemm.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "r4dx/core/r4d.hpp"

namespace r4dx::model::qwen4exp {

namespace {

constexpr int64_t kMaxChunkM = 64;  // r4d_gemm_bf16_nt_m64's row cap

// The shapes linear.cpp's FallbackTuning accepts for a bf16 weight (K % 256 == 0, N % 16 == 0).
bool ApplyLinearAccepts(int64_t N, int64_t K) { return K > 0 && K % 256 == 0 && N > 0 && N % 16 == 0; }

}  // namespace

void ApplyBf16Linear(hipStream_t stream, core::Arena& arena, const QuantLinear& w, const uint16_t* x, uint16_t* y,
                     int64_t M) {
  if (w.layout != Layout::kBf16 || ApplyLinearAccepts(w.N, w.K)) {
    ApplyLinear(stream, arena, w, x, y, M);
    return;
  }
  if (w.K % 64 != 0) {
    throw std::runtime_error("qwen4exp::ApplyBf16Linear: bf16 linear [" + std::to_string(w.N) + ", " +
                             std::to_string(w.K) + "] needs K % 64 == 0 (SK 4 x 16)");
  }
  const bool narrow = w.N <= 16;
  const int wv = narrow ? 1 : 4;
  const int sk = (narrow && w.K % 256 == 0) ? 16 : 4;
  for (int64_t m0 = 0; m0 < M; m0 += kMaxChunkM) {
    const int m = static_cast<int>(std::min(kMaxChunkM, M - m0));
    core::r4d::GemmBf16NtM64(x + m0 * w.K, w.bf16_w.data(), y + m0 * w.N, m, static_cast<int>(w.K),
                             static_cast<int>(w.N), wv, sk, /*MB=*/1, stream);
  }
}

QuantLinear MakeBf16Linear(const uint16_t* host_w, int64_t N, int64_t K) {
  QuantLinear w;
  w.layout = Layout::kBf16;
  w.N = N;
  w.K = K;
  w.bf16_w = core::DeviceBuffer<uint16_t>(static_cast<size_t>(N * K));
  w.bf16_w.CopyFromHost(host_w, static_cast<size_t>(N * K));
  return w;
}

}  // namespace r4dx::model::qwen4exp
