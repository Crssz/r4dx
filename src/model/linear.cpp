#include "linear.h"

#include <algorithm>
#include <cstdlib>
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

int EpilogueForLayout(Layout layout) {
  // Milestone 4 follow-up to docs/r9700.md R2/P2 (2026-09-20): root-caused and re-enabled for
  // w4a8/mxfp4. History: the fused epilogues (kernels.h's r4dx_epilogue: f16, fp8 row-major, int8
  // fragA8) were built and verified BYTE-EXACT in isolation (tests/kernels/test_fused_quant.cpp,
  // 135/135, M in {1,2,4,16,64} x K in {5120,6144,17408}), but wiring them into the model changed
  // w4a8/mxfp4's real generated text even though an in-model diagnostic showed zero differing
  // elements on the fused GEMMs' own outputs -- shipped disabled pending root-cause (see git log
  // on this file / docs/status.md's original R2/P2 section for the full incident writeup this pass
  // inherited).
  //
  // ROOT CAUSE (this pass): not the epilogue kernels' own math (already independently verified),
  // and not which GEMM consumed a fused buffer. It is `r4dx::core::Arena::Alloc`
  // (r4dx/core/arena.hpp): it only aligned each allocation's OWN start to the caller-requested
  // alignment, never its END. A fused epilogue's per-row fp32 scale scratch is exactly `T` floats
  // (T = chunk row count, 1..64) -- 4*T bytes, a multiple of 16 only when T%4==0 -- so at decode
  // (T=1) and MTP verify (T=2..4) the NEXT default-aligned allocation in the same layer (e.g.
  // GdnLayer::Forward's `mixed_qkv`) starts a few bytes short of 16-byte alignment, and every
  // allocation after it inherits the same drift for the rest of the layer. Every third_party/libr4d
  // kernel this arena feeds (GDN conv/kkt/chunk-scan, every quantized GEMM family, attention's
  // decode scratch) reads its operands with unchecked wide (16-byte) vector loads and silently
  // reads the wrong bytes when handed a misaligned pointer -- unlike this project's OWN
  // P6-rewritten rmsnorm/residual_rmsnorm/silu_mul kernels, which fall back to a scalar loop when
  // misaligned. Before this fusion pass every arena allocation's SIZE happened to already be a
  // multiple of 16 bytes (hidden/conv_dim/intermediate are all multiples of 8 bf16 elements), so
  // `offset_` was always incidentally 16-aligned and this was never triggered -- several call
  // sites' own comments already flagged that invariant as "incidental, not enforced" (gdn_layer.cpp,
  // attention_layer.hpp, this file's i8_scratch/fp8_scratch allocations). FIX (arena.hpp): every
  // Alloc call now rounds its own END up to 16 bytes too, so every FUTURE allocation starts
  // 16-aligned again regardless of what alignment it individually requests. Covered by a new
  // host-side ArenaAlignmentInvariant check in tests/kernels/test_fused_quant.cpp (reproduces the
  // exact odd-T scale-then-buffer allocation sequence GdnLayer/AttentionLayer/Mlp use, T=1..64, and
  // asserts every subsequent pointer is 16-aligned) and re-verified end-to-end BYTE-IDENTICAL
  // generated text (SHA-256), fusion-on vs fusion-off, for w4a8 and mxfp4 across three prompt
  // lengths x `--mtp {0,3}` (tools/validate_fusion.ps1 -- also usable as a standing regression gate;
  // docs/status.md's R2/P2 section has the full run log).
  //
  // w4a16's r4dx_epilogue_f16 is a SEPARATE, unrelated issue: it is equally CORRECT now (verified
  // the same way) but still REGRESSES decode wall-clock (-4.3%, reproducible, unchanged from the
  // original incident): r4dx_model_cast_bf16_to_f16 launches a FLAT elementwise grid
  // (blocks=ceil(M*K/256), ~20 independent workgroups at decode T=1/K=5120), while fusing it into
  // rmsnorm/residual_rmsnorm/silu_mul's own one-workgroup-per-row epilogue collapses that work onto
  // a SINGLE workgroup -- a real parallelism loss the saved launch does not cover. Per this task's
  // own instruction ("do not re-enable in that form"), w4a16 stays r4dx_epilogue_none until a
  // wide-grid f16 cast or a prefill-only fusion (dim3(rows)=dim3(T), no parallelism loss at T>1) is
  // built -- Milestone 5+ work, not part of this pass.
  //
  // R4DX_DISABLE_EPILOGUE=1 forces every layout back to r4dx_epilogue_none regardless of the mapping
  // below -- the A/B toggle tools/validate_fusion.ps1 uses to regenerate the "fusion off" baseline
  // every run from the SAME binary rather than requiring a second build. Unset in every other
  // caller (every ctest binary, every plain CLI/server invocation), which see the mapping directly.
  static const bool kDisabled = [] {
    const char* e = std::getenv("R4DX_DISABLE_EPILOGUE");
    return e != nullptr && e[0] == '1';
  }();
  if (kDisabled) return r4dx_epilogue_none;
  switch (layout) {
    case Layout::kBf16:
      return r4dx_epilogue_none;  // never quantizes its activation input -- nothing to fuse.
    case Layout::kW4a16:
      return r4dx_epilogue_none;  // Problem B: parallelism loss at decode, see comment above.
    case Layout::kW4a8:
      return r4dx_epilogue_int8_fraga8;
    case Layout::kMxfp4:
      return r4dx_epilogue_fp8_e4m3_row;
  }
  return r4dx_epilogue_none;
}

void ApplyLinear(hipStream_t stream, core::Arena& arena, const QuantLinear& w, const uint16_t* x,
                  uint16_t* y, int64_t M, const PreQuantizedActivation* pre) {
  if (w.N <= 0 || w.K <= 0) throw std::runtime_error("r4dx::model::ApplyLinear: empty weight");
  const int64_t N = w.N, K = w.K;
  const hipStream_t s = stream;

  // R2/P2 (docs/r9700.md): a caller may have already produced this call's quantized activation as
  // a fused producer epilogue (rmsnorm/residual_rmsnorm/silu_mul, kernels.h's r4dx_epilogue) --
  // when the format matches this weight's layout, skip the per-chunk quant/cast launch below
  // entirely and read straight from the caller's buffer instead of arena scratch.
  const bool have_pre = (pre != nullptr) && (pre->epilogue != r4dx_epilogue_none);
  if (have_pre && pre->epilogue != EpilogueForLayout(w.layout)) {
    throw std::runtime_error(
        "r4dx::model::ApplyLinear: PreQuantizedActivation.epilogue does not match w.layout "
        "(caller bug -- see EpilogueForLayout)");
  }

  // Per-chunk activation-quant scratch, sized for the largest chunk (<=64 rows) and reused across
  // every chunk this call makes -- one arena bump, not one per chunk. Skipped entirely when a
  // pre-quantized buffer for this layout was provided (nothing to compute).
  uint16_t* f16_scratch = nullptr;   // w4a16
  int8_t* i8_scratch = nullptr;      // w4a8
  float* i8_scale_scratch = nullptr;
  uint8_t* fp8_scratch = nullptr;    // mxfp4
  float* fp8_scale_scratch = nullptr;
  if (!have_pre) {
    switch (w.layout) {
      case Layout::kBf16:
        break;
      case Layout::kW4a16:
        f16_scratch = arena.Alloc<uint16_t>(static_cast<size_t>(kMaxChunkM * K));
        break;
      case Layout::kW4a8:
        // align_bytes=16 (review finding, 2026-09-20): r4d_gemm_w4a8's fragment read
        // (global_load_b64) reads this buffer with a wide load -- same alignment class as
        // attention_layer.hpp's own decode scratch fix. Every preceding arena allocation in a
        // layer happens to be 16-aligned today (hidden/conv_dim/intermediate are all multiples of
        // 16 at 2 bytes/element), which is incidental, not enforced -- pass it explicitly instead.
        i8_scratch = arena.Alloc<int8_t>(static_cast<size_t>(kMaxChunkM * K), /*align_bytes=*/16);
        i8_scale_scratch = arena.Alloc<float>(static_cast<size_t>(kMaxChunkM));
        break;
      case Layout::kMxfp4:
        fp8_scratch = arena.Alloc<uint8_t>(static_cast<size_t>(kMaxChunkM * K), /*align_bytes=*/16);
        fp8_scale_scratch = arena.Alloc<float>(static_cast<size_t>(kMaxChunkM));
        break;
    }
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
      case Layout::kW4a16: {
        const uint16_t* a;
        if (have_pre) {
          a = reinterpret_cast<const uint16_t*>(pre->data) + m0 * K;
        } else {
          r4dx_model_cast_bf16_to_f16(reinterpret_cast<int64_t>(xc),
                                       reinterpret_cast<int64_t>(f16_scratch),
                                       static_cast<int64_t>(m) * K, reinterpret_cast<int64_t>(s));
          a = f16_scratch;
        }
        core::r4d::GemmW4a16NtM64(a, w.wq.data(), w.w4a16_wsz.data(), yc, m, static_cast<int>(K),
                                   static_cast<int>(N), t.WV, t.SK, t.MB, t.NPW, t.NT, s);
        break;
      }
      case Layout::kW4a8: {
        const int8_t* a;
        const float* a_scale;
        if (have_pre) {
          a = reinterpret_cast<const int8_t*>(pre->data) + m0 * K;
          a_scale = pre->scale + m0;
        } else {
          core::r4d::QuantActI8(xc, i8_scratch, i8_scale_scratch, m, static_cast<int>(K), s);
          a = i8_scratch;
          a_scale = i8_scale_scratch;
        }
        core::r4d::GemmW4a8NtM64(a, a_scale, w.wq.data(), w.w4a8_ws.data(), yc, m,
                                  static_cast<int>(K), static_cast<int>(N), t.WV, t.SK, t.MB,
                                  t.NPW, t.NT, s);
        break;
      }
      case Layout::kMxfp4: {
        const uint8_t* a;
        const float* a_scale;
        if (have_pre) {
          a = reinterpret_cast<const uint8_t*>(pre->data) + m0 * K;
          a_scale = pre->scale + m0;
        } else {
          r4dx_quant_act_fp8e4m3_row(reinterpret_cast<int64_t>(xc),
                                      reinterpret_cast<int64_t>(fp8_scratch),
                                      reinterpret_cast<int64_t>(fp8_scale_scratch), m,
                                      static_cast<int>(K), reinterpret_cast<int64_t>(s));
          a = fp8_scratch;
          a_scale = fp8_scale_scratch;
        }
        core::r4d::GemmMxfp4a8NtM64(a, a_scale, w.mxfp4_wq.data(), w.mxfp4_ws.data(),
                                     w.mxfp4_wref.data(), yc, m, static_cast<int>(K),
                                     static_cast<int>(N), t.WV, t.SK, t.MB, t.NPW, s);
        break;
      }
    }
  }
}

}  // namespace r4dx::model
