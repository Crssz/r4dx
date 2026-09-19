#include "linear.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>

#include "kernels/model_kernels.h"
#include "r4d.h"
#include "r4dx/core/r4d.hpp"
#include "r4dx/kernels/kernels.h"

namespace r4dx::model {

namespace {

constexpr int64_t kMaxChunkM = 64;

// Hand-derived fallback, legal for every (layout,N,K) shape this model has (used only when
// gemm_tuning_table.inc has no row for the requested shape -- see PickTuning below and linear.h's
// comment). Every quantized GEMM family this model calls needs K divisible by SK*group() (bf16:
// group 16, w4a16/w4a8: group 128, mxfp4: group 32) and N divisible by 16. This model's only K
// values are hidden_size=5120, intermediate_size=17408, and value_dim=6144 (attn.o's K =
// num_heads*head_dim = 6144 too) -- all three are multiples of 512 (5120/512=10, 17408/512=34,
// 6144/512=12), which is the tightest of the three group requirements (SK=4 * group=128), so SK=4
// clears every layout at once. WV=4/SK=4 keeps the block at 512 threads (WV*SK*32, under the 1024
// cap every kernel enforces) and the LDS reduction buffer at 16 KiB (under the 64 KiB cap); MB=1
// and NPW=1 are the simplest legal choice for every kernel (MB in 1..4, NPW in {1,4} for w4a16 /
// {1,2,4,8} for w4a8/mxfp4) and NT=1 takes the non-temporal weight-load path
// r4d_gemm_w4a16_nt_m64.hip's own comment recommends for a weight that is read once per step and
// never reused.
LinearTuning FallbackTuning(Layout /*layout*/, int64_t N, int64_t K) {
  if (K % 512 != 0) {
    throw std::runtime_error("r4dx::model::PickTuning: K=" + std::to_string(K) +
                              " is not a multiple of 512 (SK=4 * w4a16/w4a8 group 128) -- this "
                              "model shape was not anticipated, pick a smaller SK");
  }
  if (N % 16 != 0) {
    throw std::runtime_error("r4dx::model::PickTuning: N=" + std::to_string(N) +
                              " is not a multiple of 16");
  }
  return LinearTuning{/*WV=*/4, /*SK=*/4, /*MB=*/1, /*NPW=*/1, /*NT=*/1};
}

// tools/profile/tune_gemm.py's measured sweep, if it has been generated (src/model/CMakeLists.txt
// treats a missing file as a build error -- see that file's comment -- so an empty table checked
// into the tree, `{}`, is what a fresh checkout without having run the sweep gets; PickTuning below
// falls back to FallbackTuning() row-by-row in that case, not a hard failure).
#include "gemm_tuning_table.inc"

}  // namespace

// Internal linkage (not declared in linear.h) -- the actual table scan, now called only on a
// PickTuning cache miss (see below).
static LinearTuning ResolveTuning(Layout layout, int64_t N, int64_t K, int64_t M) {
  const GemmTuningRow* best = nullptr;
  for (const GemmTuningRow& row : kGemmTuningTable) {
    if (row.layout != layout || row.N != N || row.K != K) continue;
    if (row.M < M) continue;  // only ever round UP to a wider-or-equal measured M-band
    if (best == nullptr || row.M < best->M) best = &row;
  }
  if (best != nullptr) return best->tuning;
  return FallbackTuning(layout, N, K);
}

LinearTuning PickTuning(Layout layout, int64_t N, int64_t K, int64_t M) {
  // Cache the resolved LinearTuning per (layout,N,K,M) (review finding, 2026-09-19): PickTuning is
  // called once per <=64-row sub-chunk of every GEMM -- roughly 6 GEMMs x 64 layers per decode
  // token -- and a linear scan of kGemmTuningTable's ~196 rows on every one of those calls is
  // ~75k row comparisons of pure host work per token on the exact hot path the host-overhead pass
  // (2026-09-19) was trying to minimize. Model is single-sequence / single-worker-thread
  // (model.h's own SCOPE comment; src/server also serializes all requests through one worker
  // thread onto one Model), so a plain function-local static map needs no locking. N/K fit in 20
  // bits (this model's widest is intermediate_size=17408 < 2^20), M in 8 bits (<=64), layout in 4
  // bits -- packed key never collides for any shape this model has.
  static std::unordered_map<int64_t, LinearTuning> cache;
  const int64_t key = (static_cast<int64_t>(layout) << 48) | (N << 28) | (K << 8) | M;
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  const LinearTuning t = ResolveTuning(layout, N, K, M);
  cache.emplace(key, t);
  return t;
}

void ApplyLinear(hipStream_t stream, core::Arena& arena, const QuantLinear& w, const uint16_t* x,
                  uint16_t* y, int64_t M) {
  if (w.N <= 0 || w.K <= 0) throw std::runtime_error("r4dx::model::ApplyLinear: empty weight");
  const int64_t N = w.N, K = w.K;
  const hipStream_t s = stream;

  // Per-chunk activation-quant scratch, sized for the largest chunk (<=64 rows) and reused across
  // every chunk this call makes -- one arena bump, not one per chunk.
  uint16_t* f16_scratch = nullptr;   // w4a16
  int8_t* i8_scratch = nullptr;      // w4a8
  float* i8_scale_scratch = nullptr;
  uint8_t* fp8_scratch = nullptr;    // mxfp4
  float* fp8_scale_scratch = nullptr;
  switch (w.layout) {
    case Layout::kBf16:
      break;
    case Layout::kW4a16:
      f16_scratch = arena.Alloc<uint16_t>(static_cast<size_t>(kMaxChunkM * K));
      break;
    case Layout::kW4a8:
      i8_scratch = arena.Alloc<int8_t>(static_cast<size_t>(kMaxChunkM * K));
      i8_scale_scratch = arena.Alloc<float>(static_cast<size_t>(kMaxChunkM));
      break;
    case Layout::kMxfp4:
      fp8_scratch = arena.Alloc<uint8_t>(static_cast<size_t>(kMaxChunkM * K));
      fp8_scale_scratch = arena.Alloc<float>(static_cast<size_t>(kMaxChunkM));
      break;
  }

  for (int64_t m0 = 0; m0 < M; m0 += kMaxChunkM) {
    const int m = static_cast<int>(std::min(kMaxChunkM, M - m0));
    const uint16_t* xc = x + m0 * K;
    uint16_t* yc = y + m0 * N;
    // Picked per sub-chunk (not once for the whole call): a decode call (M=1) and a prefill call
    // (M up to 64, chunked here into <=64-row slices) want different WV/SK/MB/NPW even for the
    // same (layout,N,K) -- PickTuning's table is keyed by the exact per-launch row count `m`, not
    // the caller's total `M` (see linear.h's PickTuning comment on M-band rounding).
    const LinearTuning t = PickTuning(w.layout, N, K, m);

    switch (w.layout) {
      case Layout::kBf16:
        core::r4d::GemmBf16NtM64(xc, w.bf16_w.data(), yc, m, static_cast<int>(K),
                                  static_cast<int>(N), t.WV, t.SK, t.MB, s);
        break;
      case Layout::kW4a16:
        r4dx_model_cast_bf16_to_f16(reinterpret_cast<int64_t>(xc),
                                     reinterpret_cast<int64_t>(f16_scratch),
                                     static_cast<int64_t>(m) * K, reinterpret_cast<int64_t>(s));
        core::r4d::GemmW4a16NtM64(f16_scratch, w.wq.data(), w.w4a16_wsz.data(), yc, m,
                                   static_cast<int>(K), static_cast<int>(N), t.WV, t.SK, t.MB,
                                   t.NPW, t.NT, s);
        break;
      case Layout::kW4a8:
        core::r4d::QuantActI8(xc, i8_scratch, i8_scale_scratch, m, static_cast<int>(K), s);
        core::r4d::GemmW4a8NtM64(i8_scratch, i8_scale_scratch, w.wq.data(), w.w4a8_ws.data(), yc, m,
                                  static_cast<int>(K), static_cast<int>(N), t.WV, t.SK, t.MB,
                                  t.NPW, t.NT, s);
        break;
      case Layout::kMxfp4:
        r4dx_quant_act_fp8e4m3_row(reinterpret_cast<int64_t>(xc),
                                    reinterpret_cast<int64_t>(fp8_scratch),
                                    reinterpret_cast<int64_t>(fp8_scale_scratch), m,
                                    static_cast<int>(K), reinterpret_cast<int64_t>(s));
        core::r4d::GemmMxfp4a8NtM64(fp8_scratch, fp8_scale_scratch, w.mxfp4_wq.data(),
                                     w.mxfp4_ws.data(), w.mxfp4_wref.data(), yc, m,
                                     static_cast<int>(K), static_cast<int>(N), t.WV, t.SK, t.MB,
                                     t.NPW, s);
        break;
    }
  }
}

}  // namespace r4dx::model
