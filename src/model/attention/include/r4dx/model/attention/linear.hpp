// r4dx::model::attention::Linear -- a minimal bf16 GEMM wrapper over r4d_gemm_bf16_nt_m64 for
// this component's attn.k / attn.v projections, which have no quantized on-disk form
// (docs/container-format.md: "attn.k/v (bf16-only, no .{layout} suffix)") and so are always bf16
// regardless of --layout. attn.qg/attn.o now dispatch through the shared r4dx::model::ApplyLinear
// (src/model/linear.h, decode-perf pass 2026-09-19) instead -- see attention_layer.hpp -- since
// those two DO have quantized on-disk forms and honoring `--layout` for them is the point of that
// pass ("dedupe the duplicated linear logic between core and attention" per the task brief this
// component's original scope anticipated).
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>

#include "r4dx/core/r4d.hpp"

namespace r4dx::model::attention {

struct GemmTuning {
  int WV, SK, MB;
};

// Picks a valid WV/SK/MB for r4d_gemm_bf16_nt_m64 (K % (SK*16) == 0, WV*SK*32 <= 1024;
// third_party/libr4d/r4d_gemm_bf16_nt_m64.hip) for a given K. "The model graph picks WV/SK/MB per
// shape once at layer-build time" per docs/architecture.md -- this is that policy, not tuned for
// throughput (that is the dedicated prefill-kernel milestone's job; docs/architecture.md "Interim
// chunked prefill"), only guaranteed VALID for every K this layer uses (hidden=5120 for
// qg/k/v, num_heads*head_dim=6144 for o_proj).
inline GemmTuning ChooseBf16NtM64Tuning(int K) {
  if (K <= 0 || K % 16 != 0) {
    throw std::invalid_argument("ChooseBf16NtM64Tuning: K must be a positive multiple of 16");
  }
  const int k16 = K / 16;
  for (int sk : {16, 8, 4, 2, 1}) {
    if (k16 % sk == 0) {
      return GemmTuning{32 / sk, sk, 1};
    }
  }
  return GemmTuning{1, 1, 1};  // unreachable: sk=1 always divides k16
}

// C[M,N] = A[M,K] @ W[N,K]^T, bf16 throughout. M is bounded by r4d_gemm_bf16_nt_m64's own 1..64
// skinny-GEMM band (docs/architecture.md: "At M<=64 ... every GEMM above is one of the skinny
// r4d_gemm_*_nt_m64 kernels").
struct Linear {
  const uint16_t* w = nullptr;  // device, [N, K] bf16, row-major
  int N = 0, K = 0;

  void Gemm(const uint16_t* a, int M, uint16_t* c, hipStream_t stream) const {
    if (M < 1 || M > 64) {
      throw std::invalid_argument("Linear::Gemm: M must be 1..64 (r4d_gemm_bf16_nt_m64 band)");
    }
    const GemmTuning t = ChooseBf16NtM64Tuning(K);
    r4dx::core::r4d::GemmBf16NtM64(a, w, c, M, K, N, t.WV, t.SK, t.MB, stream);
  }
};

}  // namespace r4dx::model::attention
