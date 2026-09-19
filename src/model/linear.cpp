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

int EpilogueForLayout(Layout layout) {
  // NOT WIRED (docs/r9700.md R2/P2, 2026-09-20) -- returns r4dx_epilogue_none for every layout,
  // deliberately, pending a fix. Full status:
  //
  // The fused epilogues themselves (kernels.h's r4dx_epilogue: f16, fp8 row-major, int8 fragA8)
  // are implemented in src/kernels/src/r4dx_kernels.hip and verified BYTE-EXACT against the
  // standalone kernels they'd replace (r4dx_model_cast_bf16_to_f16 / r4dx_quant_act_fp8e4m3_row /
  // third_party/libr4d's r4d_quant_act_i8) by tests/kernels/test_fused_quant.cpp, 135/135 checks,
  // M in {1,2,4,16,64} x K in {5120,6144,17408} -- the task's own full grid, in isolation.
  //
  // w4a16's r4dx_epilogue_f16 was additionally found, when wired, to REGRESS decode wall-clock
  // (-4.3%, reproducible): r4dx_model_cast_bf16_to_f16 launches a FLAT elementwise grid
  // (blocks=ceil(M*K/256), ~20 independent workgroups at decode T=1/K=5120), while fusing it into
  // rmsnorm/residual_rmsnorm/silu_mul's own one-workgroup-per-row epilogue collapses that work onto
  // a SINGLE workgroup -- a real parallelism loss the saved launch does not cover. This layout's
  // fusion is deliberately not re-enabled even once the item below is fixed; see git history on
  // this file (or docs/status.md's R2/P2 section) for the measured numbers and the "fuse only
  // during prefill" follow-up idea.
  //
  // w4a8's r4dx_epilogue_int8_fraga8 and mxfp4's r4dx_epilogue_fp8_e4m3_row, when wired, were found
  // to change the model's generated token stream on real hardware (confirmed via a real -- not
  // isolated -- CLI generation, reproducible across runs, not GPU nondeterminism: with all fusion
  // disabled the same prompt/flags reproduce byte-identical text across repeated runs) even though
  // an in-model diagnostic (temporarily comparing each fused GEMM's own output against a second,
  // unfused ApplyLinear call on the SAME inputs, immediately after the fused call, for
  // gdn.in_proj_qkv/gdn.in_proj_z/mlp.down) showed ZERO differing elements across 200+ real
  // decode-step samples spanning many layers. The root cause is therefore NOT in the epilogue
  // kernels' own math (independently verified twice, in isolation and in-model) and was not
  // isolated within this pass's time budget; likely candidate areas for a follow-up investigation,
  // in rough priority order: (1) the NEW per-call arena allocations this fusion adds (epilogue
  // scratch, allocated between existing allocations in GdnLayer/AttentionLayer/Mlp) shifting every
  // LATER allocation in the same layer to a different offset than the pre-fusion code path used,
  // interacting badly with something that is sensitive to absolute arena layout rather than going
  // through Arena::Alloc's own bump-then-return contract; (2) gate_up's OWN local fused epilogue
  // (mlp.cpp, x_normed_in==nullptr path) specifically -- unlike gdn.in_proj_qkv/z/mlp.down, it was
  // not itself isolated with the same in-model diagnostic before this pass's time ran out; (3) the
  // qg/k/v attention-layer fusion path (attention_layer.hpp) -- never exercised by this bisection
  // at all (layer 0 in this container is a GDN layer, and cross-layer-boundary fusion was disabled
  // throughout the bisection), so it is UNTESTED, not cleared. Per this task's own explicit
  // instruction ("never loosen to a tolerance... shipping it unverified risks silently corrupting
  // every quantized GEMM's input, which is worse than not shipping it"), this returns
  // r4dx_epilogue_none unconditionally until the root cause is found and a fix is verified the same
  // way test_fused_quant.cpp already verifies the kernels: real generated text, byte-identical to
  // the unfused baseline, not just a tolerance-bounded numeric check.
  (void)layout;
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
