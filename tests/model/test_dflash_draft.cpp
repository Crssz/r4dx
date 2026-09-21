// tests/model/test_dflash_draft.cpp -- Milestone 5 stage S2 item 3: the whole DFlash2 draft module
// (src/model/dflash_draft.{h,cpp}) end to end on device, against the Python reference's golden
// fixtures (docs/dflash2.md section 9).
//
// WHAT IS CHECKED, and against what
// ---------------------------------
// The reference (`tools/reference/dflash2_ref.py`) runs the SAME real draft weights this test
// loads, but in fp32/fp64 numpy against a synthetic target (vocab 4096, a seeded-random embedding
// table and lm_head). So this test reproduces that exact synthetic setup on the GPU -- the two
// injectable providers `DflashDraft::DraftRound` takes exist precisely so a test can swap the real
// 27B target for that 4096-row synthetic one -- and compares every intermediate the fixture stores:
//
//   Part 0  The numpy-stream reimplementation (numpy_legacy_rng.hpp) is validated BIT-EXACT against
//           fixture A's own dumped `features.npy` before it is trusted for the two inputs the
//           fixtures deliberately do NOT dump (the synthetic target's tables, fixture B's features).
//   Part 1  Fixture A, bf16 draft container: `g` (encoder), per-layer injected K and V, per-layer
//           `x` after attention and after FFN, `x_final_normed`, `logits`, `cand`, `unary`, `gate`,
//           every `score_t{1..7}` matrix, and the drafted chain (which must match EXACTLY).
//   Part 2  Fixture C: the same injected state re-drafted with `p_min=0.3` -- the selector's early
//           stop. Drafted chain must match exactly (5 tokens, a strict prefix of A's 7).
//   Part 3  Fixture B: `n_injected=2100`, which forces the 2048-wide sliding window to clip and the
//           KV ring to wrap. Drafted chain must match exactly.
//   Part 4  The w4a16 draft container on fixture A's inputs: cand overlap and chain agreement
//           against the bf16 run. Drift is EXPECTED here (4-bit weights); this part quantifies it
//           rather than gating it.
//
// TOLERANCES. Every gate is on `RelL2` (relative L2 over the whole tensor), never per-element
// max_rel: the reference accumulates in float64 while this path is bf16-in/bf16-out with fp32
// GEMM accumulation, so an element that cancels to near zero disagrees by ~100% of its own
// magnitude no matter how correct the kernel is. The thresholds below were set from the MEASURED
// values (printed by every check, so a regression shows its own number) with ~2x headroom; they are
// still sharp, because a wrong layout/mask/tap/side index moves RelL2 by two or more orders of
// magnitude, not by a factor of two.
//
// SKIPs (77, CTest SKIPPED) when the draft containers or the golden fixtures are absent -- both are
// regenerable and neither is vendored (docs/dflash2.md sections 9 and 11).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "dflash_draft.h"
#include "linear.h"
#include "nlohmann/json.hpp"
#include "npy_fixture.hpp"  // tests/kernels/, shared via an include dir (see this dir's CMakeLists)
#include "numpy_legacy_rng.hpp"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/stream.hpp"
#include "r4dx/kernels/kernels.h"
#include "kernels/model_kernels.h"
// NOTE: deliberately NOT including tests/model/test_common.h. Both it and tests/kernels/
// npy_fixture.hpp define `r4dx_test::kSkipReturnCode`, so including both in one TU is a
// redefinition error; npy_fixture.hpp is the one this test genuinely needs (the .npy reader), and
// the three helpers test_common.h would have provided are four lines each, below.

using r4dx::core::Arena;
using r4dx::core::Bf16ToFloat;
using r4dx::core::DeviceBuffer;
using r4dx::core::FloatToBf16;
using r4dx::core::Stream;
using r4dx::model::ApplyLinear;
using r4dx::model::DflashDraft;
using r4dx::model::DflashDraftOptions;
using r4dx::model::DflashDraftResult;
using r4dx::model::DflashEmbeddingProvider;
using r4dx::model::DflashLmHeadProvider;
using r4dx::model::DflashRoundTrace;
using r4dx::model::Layout;
using r4dx::model::QuantLinear;

namespace {

const char* kDraftBf16 = "D:/models/r4dx/qwen38-27b-dflash2-bf16.r4dx";
const char* kDraftW4a16 = "D:/models/r4dx/qwen38-27b-dflash2-w4a16.r4dx";

int g_failures = 0;

bool FileExists(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return static_cast<bool>(f);
}

int SkipMissing(const std::string& what) {
  std::fprintf(stderr,
               "[SKIP] %s not found; this test needs the DFlash2 draft container plus the "
               "regenerable golden fixtures -- see docs/dflash2.md sections 9 and 11.\n",
               what.c_str());
  return r4dx_test::kSkipReturnCode;
}

std::vector<float> WidenBf16(const std::vector<uint16_t>& x) {
  std::vector<float> out(x.size());
  for (size_t i = 0; i < x.size(); ++i) out[i] = Bf16ToFloat(x[i]);
  return out;
}

// sqrt(sum((got-ref)^2)) / sqrt(sum(ref^2)) -- the same metric npy_fixture.hpp's ErrStats::norm_rel
// and tests/model/test_common.h's RelL2 both use.
double RelL2(const std::vector<float>& got, const std::vector<float>& ref) {
  return r4dx_test::CompareF32ToRef(got, ref).norm_rel;
}

// Gates. Every one of these is a MEASURED value from the first green run, rounded up ~2x.
constexpr double kTolG = 6e-3;          // encoder output
constexpr double kTolInjectK = 1e-2;    // injected K (GEMM + per-head norm + rope)
constexpr double kTolInjectV = 6e-3;    // injected V (GEMM only)
constexpr double kTolLayerX = 4e-2;     // per-layer residual, worst (last) layer
constexpr double kTolFinal = 4e-2;      // x_final_normed
constexpr double kTolLogits = 6e-2;     // logits (adds the synthetic bf16 lm_head GEMM)
constexpr double kTolUnary = 6e-2;      // == top-16 of logits
constexpr double kTolGate = 4e-2;       // selector gate GEMM
constexpr double kTolScore = 1e-1;      // score = codebook trilinear form over `gate` + unary
// Fixture B only. At n_injected=2100 the block's attention averages over ~2048 visible keys instead
// of 40, and a diffuse softmax's output is a heavily cancelling sum: the same per-element bf16 input
// error therefore lands ~3x larger, relative, on `x_final` and so on `gate` (measured 1.20e-1 vs
// fixture A's 3.33e-2 on the identical code path). This is NOT the windowing: the attention kernel
// itself was swept against a CPU fp64 reference at n_injected in {0,3,40,2047,2048,2049,2100,5000}
// and stayed pinned at 1.6e-3 throughout, ring wrap and window clip included
// (tests/kernels/test_dflash_attn.cpp, docs/dflash2.md section 6b). The properties that would
// actually catch a windowing bug -- cand/unary bit-exactness and the drafted chain -- are gated
// exactly for fixture B like every other fixture, and pass.
constexpr double kTolGateWideWindow = 2.5e-1;

void Check(bool ok, const char* what, double got, double tol) {
  std::printf("  %-34s RelL2=%.3e  (tol %.1e)  %s\n", what, got, tol, ok ? "[PASS]" : "[FAIL]");
  if (!ok) ++g_failures;
}

void CheckRel(const std::vector<float>& got, const std::vector<float>& ref, const char* what,
              double tol) {
  if (got.size() != ref.size()) {
    std::printf("  %-34s SIZE MISMATCH got=%zu ref=%zu [FAIL]\n", what, got.size(), ref.size());
    ++g_failures;
    return;
  }
  const double r = RelL2(got, ref);
  Check(r <= tol, what, r, tol);
}

void Fail(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  std::vfprintf(stdout, fmt, ap);
  va_end(ap);
  ++g_failures;
}

std::vector<uint16_t> ToBf16Vec(const std::vector<float>& x) {
  std::vector<uint16_t> out(x.size());
  for (size_t i = 0; i < x.size(); ++i) out[i] = FloatToBf16(x[i]);
  return out;
}

nlohmann::json LoadManifest(const std::string& fixture) {
  const std::string p = r4dx_test::FixtureDir(fixture) + "/manifest.json";
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + p);
  nlohmann::json j;
  f >> j;
  return j;
}

std::string FixPath(const std::string& fixture, const std::string& name) {
  return r4dx_test::FixtureDir(fixture) + "/" + name + ".npy";
}

// The synthetic target the reference's `SyntheticTarget` builds: embedding table drawn first from
// RandomState(seed), lm_head drawn immediately after from the SAME instance, both scaled by 0.02 in
// float64 and then cast to float32 (dflash2_ref.py lines 271-278).
struct SyntheticTarget {
  int64_t vocab = 0, hidden = 0;
  DeviceBuffer<uint16_t> embed_dev;  // [vocab, hidden] bf16
  QuantLinear lm_head;               // bf16 [vocab, hidden]

  void Build(int64_t v, int64_t h, uint32_t seed) {
    vocab = v;
    hidden = h;
    const size_t n = static_cast<size_t>(v * h);
    r4dx_test::NumpyLegacyRng rng(seed);
    std::vector<float> tmp(n);
    rng.FillRandnScaledF32(tmp.data(), n, 0.02);
    {
      std::vector<uint16_t> e = ToBf16Vec(tmp);
      embed_dev = DeviceBuffer<uint16_t>(n);
      embed_dev.CopyFromHost(e);
    }
    rng.FillRandnScaledF32(tmp.data(), n, 0.02);
    {
      std::vector<uint16_t> w = ToBf16Vec(tmp);
      lm_head.layout = Layout::kBf16;
      lm_head.N = v;
      lm_head.K = h;
      lm_head.bf16_w = DeviceBuffer<uint16_t>(n);
      lm_head.bf16_w.CopyFromHost(w);
    }
  }

  DflashEmbeddingProvider EmbedProvider() const {
    const uint16_t* table = embed_dev.data();
    const int64_t h = hidden, v = vocab;
    return [table, h, v](Stream& stream, const int32_t*, const int32_t* ids_dev, int64_t n,
                         uint16_t* out_dev) {
      r4dx_embedding_gather_bf16(reinterpret_cast<int64_t>(table),
                                 reinterpret_cast<int64_t>(ids_dev),
                                 reinterpret_cast<int64_t>(out_dev), n, h, v,
                                 reinterpret_cast<int64_t>(stream.get()));
    };
  }

  DflashLmHeadProvider LmHeadProvider() const {
    const QuantLinear* w = &lm_head;
    return [w](Stream& stream, Arena& arena, const uint16_t* x_dev, float* logits_out, int64_t T) {
      uint16_t* bf = arena.Alloc<uint16_t>(static_cast<size_t>(T * w->N));
      ApplyLinear(stream, arena, *w, x_dev, bf, T);
      r4dx_model_widen_bf16_to_f32(reinterpret_cast<int64_t>(bf),
                                   reinterpret_cast<int64_t>(logits_out), T * w->N,
                                   reinterpret_cast<int64_t>(stream.get()));
    };
  }
};

// Uploads `rows` feature rows (fp32 host -> bf16 device) and injects them in <=64-row pieces,
// exactly as a real driver drains Model's own capture buffer chunk by chunk. `base_pos` shifts the
// whole run to a higher absolute position, which Part 5 uses to open a cold-ring gap.
void InjectAll(DflashDraft& d, Stream& stream, Arena& arena, const std::vector<float>& features_f32,
               int64_t rows, int64_t cols, int64_t base_pos = 0) {
  DeviceBuffer<uint16_t> stage(static_cast<size_t>(64 * cols));
  std::vector<uint16_t> host(static_cast<size_t>(64 * cols));
  for (int64_t off = 0; off < rows; off += 64) {
    const int64_t n = std::min<int64_t>(64, rows - off);
    for (int64_t i = 0; i < n * cols; ++i) {
      host[static_cast<size_t>(i)] = FloatToBf16(features_f32[static_cast<size_t>(off * cols + i)]);
    }
    stage.CopyFromHost(host.data(), static_cast<size_t>(n * cols));
    d.InjectFeatures(stream, arena, stage.data(), n, base_pos + off);
    stream.Synchronize();
    arena.Reset();
  }
}

// Prints and returns the drafted chain's agreement with a reference list.
bool CompareChain(const std::vector<int32_t>& got, const std::vector<int64_t>& ref,
                  const char* what) {
  bool ok = got.size() == ref.size();
  if (ok) {
    for (size_t i = 0; i < got.size(); ++i) {
      if (static_cast<int64_t>(got[i]) != ref[i]) ok = false;
    }
  }
  std::printf("  %-34s got=[", what);
  for (size_t i = 0; i < got.size(); ++i) std::printf("%s%d", i ? "," : "", got[i]);
  std::printf("] ref=[");
  for (size_t i = 0; i < ref.size(); ++i) std::printf("%s%lld", i ? "," : "", (long long)ref[i]);
  std::printf("] %s\n", ok ? "[PASS]" : "[FAIL]");
  if (!ok) ++g_failures;
  return ok;
}

// When a chain DOES diverge, this prints the selector margin at every walked position -- the
// top1-top2 gap of exactly the row the walk read (row `pred_idx`, which is 0 at t==1 and the
// PREVIOUS position's chosen column afterwards) -- so a reader can tell "a near-tie flipped" from
// "the port is wrong" (this test's own requirement).
void PrintWalkMargins(const DflashRoundTrace& tr) {
  const size_t topk = 16;
  size_t pred_idx = 0;
  for (size_t t = 0; t < tr.score.size(); ++t) {
    const std::vector<float>& mat = tr.score[t];
    if (mat.empty() || tr.walk_b[t] < 0) break;
    const size_t rows = mat.size() / topk;
    const size_t a = std::min(pred_idx, rows - 1);
    float best = -1e30f, second = -1e30f;
    for (size_t j = 0; j < topk; ++j) {
      const float v = mat[a * topk + j];
      if (v > best) {
        second = best;
        best = v;
      } else if (v > second) {
        second = v;
      }
    }
    std::printf("    t=%zu pred_idx=%zu top1-top2 margin=%.4e  prob@argmax=%.4f  b=%d\n", t + 1, a,
                best - second, tr.walk_prob[t], tr.walk_b[t]);
    pred_idx = static_cast<size_t>(tr.walk_b[t]);
  }
}

// An LmHeadProvider that ignores `x_dev` and hands back a FIXED [block_size, vocab] fp32 logits
// matrix -- the reference fixture's own `logits.npy`.
//
// WHY THIS EXISTS, and why it is the primary gate rather than a shortcut. The fixtures' target is
// SYNTHETIC: `lm_head = RandomState(0).randn(4096, 5120) * 0.02` applied to a ~unit-RMS hidden
// state, so the 4096 logits of a row are near-Gaussian and the gaps between the 16th and 17th order
// statistics are ~0.05 sigma. This path's own logits agree with the reference to ~4e-2 relative
// (measured, printed below) -- which is genuinely good for a bf16 pipeline, and still larger than
// those gaps, so the top-16 SET and especially its ORDER reshuffle. That matters far more than it
// looks: the walk's `pred_idx` is an index INTO the previous position's candidate list, so a single
// reordering at position t silently re-points position t+1's whole score row at a different token.
// It is a property of the synthetic target, not of the drafter.
//
// Feeding the reference's exact logits removes that one source of noise and leaves everything this
// module actually owns under test: the top-16 (`r4dx_topk16_f32`, already proven EXACT on these
// very logits by tests/kernels/test_topk16.cpp), the selector-gate GEMM, the codebook trilinear
// form, the greedy walk, p_min and n_min. With cand identical, the `score_t{1..7}` matrices are
// column-aligned with the fixture's and can be compared at all. The full synthetic-lm_head path is
// still exercised right afterwards and its drift reported -- it is just not the exactness gate.
DflashLmHeadProvider FixedLogitsProvider(const std::vector<float>& logits, int64_t vocab,
                                         std::shared_ptr<DeviceBuffer<float>> dev) {
  dev->Resize(logits.size());
  dev->CopyFromHost(logits);
  return [dev, vocab](Stream& stream, Arena&, const uint16_t*, float* out, int64_t T) {
    R4DX_HIP_CHECK(hipMemcpyAsync(out, dev->data(), static_cast<size_t>(T * vocab) * sizeof(float),
                                  hipMemcpyDeviceToDevice, stream.get()));
  };
}

// The exactness gate shared by all three fixtures: run one round on the drafter's CURRENT injected
// state with the fixture's own logits substituted for the target lm_head, then check cand/unary
// bit-exactly, gate and every score matrix by RelL2, and the drafted chain exactly.
DflashDraftResult CheckSelector(DflashDraft& d, Stream& stream, Arena& arena,
                                const std::string& fixture, int32_t anchor, float p_min, int64_t B,
                                int64_t vocab, const DflashEmbeddingProvider& embed,
                                double gate_tol = kTolGate) {
  const std::vector<float> ref_logits =
      r4dx_test::LoadNpyF32(FixPath(fixture, "logits"), {B, vocab});
  auto dev = std::make_shared<DeviceBuffer<float>>();
  const DflashLmHeadProvider provider = FixedLogitsProvider(ref_logits, vocab, dev);

  DflashRoundTrace tr;
  DflashDraftResult res =
      d.DraftRound(stream, arena, anchor, B - 1, p_min, /*n_min=*/0, embed, provider, &tr);
  arena.Reset();

  const std::vector<int64_t> ref_cand = r4dx_test::LoadNpyI64(FixPath(fixture, "cand"), {B, 16});
  int64_t cand_bad = 0;
  for (int64_t i = 0; i < B * 16; ++i) {
    if (static_cast<int64_t>(tr.cand[static_cast<size_t>(i)]) != ref_cand[static_cast<size_t>(i)]) {
      ++cand_bad;
    }
  }
  std::printf("  %-34s mismatches=%lld / %lld  %s\n", "cand (exact)", (long long)cand_bad,
              (long long)(B * 16), cand_bad == 0 ? "[PASS]" : "[FAIL]");
  if (cand_bad != 0) ++g_failures;

  const std::vector<float> ref_unary = r4dx_test::LoadNpyF32(FixPath(fixture, "unary"), {B, 16});
  int64_t unary_bad = 0;
  for (int64_t i = 0; i < B * 16; ++i) {
    if (tr.unary[static_cast<size_t>(i)] != ref_unary[static_cast<size_t>(i)]) ++unary_bad;
  }
  std::printf("  %-34s mismatches=%lld / %lld  %s\n", "unary (exact)", (long long)unary_bad,
              (long long)(B * 16), unary_bad == 0 ? "[PASS]" : "[FAIL]");
  if (unary_bad != 0) ++g_failures;

  CheckRel(tr.gate, r4dx_test::LoadNpyF32(FixPath(fixture, "gate"), {B, 256}), "gate", gate_tol);
  for (int64_t t = 1; t < B; ++t) {
    const std::string nm = "score_t" + std::to_string(t);
    if (tr.score[static_cast<size_t>(t - 1)].empty()) break;  // walk stopped early (fixture C)
    CheckRel(tr.score[static_cast<size_t>(t - 1)],
             r4dx_test::LoadNpyF32(FixPath(fixture, nm), r4dx_test::NpyShape(FixPath(fixture, nm))),
             nm.c_str(), kTolScore);
  }
  const std::vector<int64_t> ref_tok = r4dx_test::LoadNpyI64(FixPath(fixture, "drafted_tokens"));
  if (!CompareChain(res.tokens, ref_tok, "drafted chain (exact)")) PrintWalkMargins(tr);
  return res;
}

// The full end-to-end path (the drafter's own x_final through a real bf16 lm_head GEMM), reported
// rather than gated -- see FixedLogitsProvider's comment for why the synthetic target's logits
// cannot be expected to preserve the top-16 ORDER through a bf16 pipeline.
void ReportEndToEnd(DflashDraft& d, Stream& stream, Arena& arena, const std::string& fixture,
                    int32_t anchor, float p_min, int64_t B, int64_t vocab,
                    const DflashEmbeddingProvider& embed, const DflashLmHeadProvider& lm_head,
                    DflashRoundTrace* trace_out) {
  DflashRoundTrace tr;
  const DflashDraftResult res =
      d.DraftRound(stream, arena, anchor, B - 1, p_min, 0, embed, lm_head, &tr);
  arena.Reset();
  const std::vector<float> ref_logits =
      r4dx_test::LoadNpyF32(FixPath(fixture, "logits"), {B, vocab});
  const std::vector<int64_t> ref_cand = r4dx_test::LoadNpyI64(FixPath(fixture, "cand"), {B, 16});
  const std::vector<int64_t> ref_tok = r4dx_test::LoadNpyI64(FixPath(fixture, "drafted_tokens"));
  int64_t exact = 0, overlap = 0;
  for (int64_t i = 0; i < B * 16; ++i) {
    if (static_cast<int64_t>(tr.cand[static_cast<size_t>(i)]) == ref_cand[static_cast<size_t>(i)]) {
      ++exact;
    }
    const int64_t t = i / 16;
    for (int64_t j2 = 0; j2 < 16; ++j2) {
      if (static_cast<int64_t>(tr.cand[static_cast<size_t>(i)]) ==
          ref_cand[static_cast<size_t>(t * 16 + j2)]) {
        ++overlap;
        break;
      }
    }
  }
  size_t agree = 0;
  while (agree < res.tokens.size() && agree < ref_tok.size() &&
         static_cast<int64_t>(res.tokens[agree]) == ref_tok[agree]) {
    ++agree;
  }
  std::printf("  end-to-end (synthetic bf16 lm_head, reported not gated):\n");
  std::printf("    logits RelL2=%.3e  cand exact-position %lld/%lld, set-overlap %lld/%lld\n",
              RelL2(tr.logits, ref_logits), (long long)exact, (long long)(B * 16),
              (long long)overlap, (long long)(B * 16));
  std::printf("    chain=[");
  for (size_t i = 0; i < res.tokens.size(); ++i) std::printf("%s%d", i ? "," : "", res.tokens[i]);
  std::printf("]  agreeing prefix vs fixture = %zu/%zu\n", agree, ref_tok.size());
  if (agree < ref_tok.size()) PrintWalkMargins(tr);
  for (int32_t id : res.tokens) {
    if (id < 0 || id >= vocab) Fail("FAIL: drafted an out-of-vocab id %d\n", id);
  }
  if (trace_out != nullptr) *trace_out = std::move(tr);
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);

  if (!FileExists(kDraftBf16)) return SkipMissing(kDraftBf16);
  if (!r4dx_test::FixtureAvailable("fixture_a") || !r4dx_test::FixtureAvailable("fixture_b") ||
      !r4dx_test::FixtureAvailable("fixture_c")) {
    return SkipMissing("tools/reference/golden_out/dflash2/fixture_{a,b,c}");
  }
  R4DX_HIP_CHECK(hipSetDevice(0));  // HIP_VISIBLE_DEVICES=1 remaps physical device 1 to index 0

  const nlohmann::json man_a = LoadManifest("fixture_a");
  const nlohmann::json man_b = LoadManifest("fixture_b");
  const nlohmann::json man_c = LoadManifest("fixture_c");
  const int64_t vocab = man_a.at("vocab").get<int64_t>();
  const int64_t mask_id = man_a.at("mask_id").get<int64_t>();
  const int32_t anchor_a = man_a.at("anchor_id").get<int32_t>();
  const int32_t anchor_b = man_b.at("anchor_id").get<int32_t>();
  const int32_t anchor_c = man_c.at("anchor_id").get<int32_t>();
  const int64_t n_inj_a = man_a.at("n_injected").get<int64_t>();
  const int64_t n_inj_b = man_b.at("n_injected").get<int64_t>();
  const float p_min_c = man_c.at("p_min").get<float>();

  // ---- Part 0: validate the numpy-stream reimplementation ---------------------------------------
  std::printf("[Part 0] numpy RandomState(0).randn bit-exactness vs fixture A's features.npy\n");
  const int64_t feat_cols = 25600;
  const std::vector<float> feat_a =
      r4dx_test::LoadNpyF32(FixPath("fixture_a", "features"), {n_inj_a, feat_cols});
  {
    r4dx_test::NumpyLegacyRng rng(0);
    std::vector<float> regen(feat_a.size());
    rng.FillRandnF32(regen.data(), regen.size());
    size_t bad = 0;
    for (size_t i = 0; i < regen.size(); ++i) {
      if (regen[i] != feat_a[i]) ++bad;
    }
    std::printf("  %-34s mismatches=%zu / %zu  %s\n", "features regenerated", bad, regen.size(),
                bad == 0 ? "[PASS]" : "[FAIL]");
    if (bad != 0) {
      ++g_failures;
      std::printf("  numpy stream reimplementation is wrong -- everything below that depends on a\n"
                  "  non-dumped synthetic input (the target tables, fixture B's features) is\n"
                  "  meaningless, so stopping here.\n");
      return 1;
    }
  }

  const int64_t hidden = 5120;
  SyntheticTarget target;
  target.Build(vocab, hidden, 0);
  const DflashEmbeddingProvider embed = target.EmbedProvider();
  const DflashLmHeadProvider lm_head = target.LmHeadProvider();

  Stream stream;
  Arena arena;
  arena.Reserve(256ull * 1024 * 1024);

  DflashDraftOptions opts;
  opts.container_path = kDraftBf16;
  opts.layout = Layout::kBf16;
  opts.lm_head_vocab = vocab;
  opts.mask_token_id_override = mask_id;
  DflashDraft draft = DflashDraft::Load(opts);

  const int64_t n_layer = draft.Config().block_count;
  const int64_t B = draft.BlockSize();
  if (draft.FeatureCols() != feat_cols) {
    Fail("FAIL: FeatureCols()=%lld, expected %lld\n", (long long)draft.FeatureCols(),
         (long long)feat_cols);
    return 1;
  }

  // ---- Part 1: fixture A -------------------------------------------------------------------------
  std::printf("[Part 1] fixture A (bf16 draft container, N=%lld injected)\n", (long long)n_inj_a);
  InjectAll(draft, stream, arena, feat_a, n_inj_a, feat_cols);
  if (draft.InjectedCount() != n_inj_a) {
    Fail("FAIL: InjectedCount()=%lld after injecting %lld rows\n",
         (long long)draft.InjectedCount(), (long long)n_inj_a);
  }
  {
    const std::vector<float> ref_g =
        r4dx_test::LoadNpyF32(FixPath("fixture_a", "g_encoded"), {n_inj_a, hidden});
    CheckRel(WidenBf16(draft.DebugEncodedG(stream, n_inj_a)), ref_g, "g = enc(fc(feat))",
             kTolG);
  }
  for (int64_t il = 0; il < n_layer; ++il) {
    const std::vector<float> rk = r4dx_test::LoadNpyF32(
        FixPath("fixture_a", "injected_k_l" + std::to_string(il)),
        {n_inj_a, draft.Config().attention.head_count_kv, draft.Config().attention.key_length});
    const std::vector<float> rv = r4dx_test::LoadNpyF32(
        FixPath("fixture_a", "injected_v_l" + std::to_string(il)),
        {n_inj_a, draft.Config().attention.head_count_kv, draft.Config().attention.key_length});
    CheckRel(WidenBf16(draft.DebugStoreK(stream, il, 0, n_inj_a)), rk,
             ("injected K l" + std::to_string(il)).c_str(), kTolInjectK);
    CheckRel(WidenBf16(draft.DebugStoreV(stream, il, 0, n_inj_a)), rv,
             ("injected V l" + std::to_string(il)).c_str(), kTolInjectV);
  }

  DflashRoundTrace tr_a;
  const DflashDraftResult res_a =
      draft.DraftRound(stream, arena, anchor_a, B - 1, /*p_min=*/0.0f, /*n_min=*/0, embed,
                       FixedLogitsProvider(r4dx_test::LoadNpyF32(FixPath("fixture_a", "logits"),
                                                                 {B, vocab}),
                                           vocab, std::make_shared<DeviceBuffer<float>>()),
                       &tr_a);
  arena.Reset();
  for (int64_t il = 0; il < n_layer; ++il) {
    CheckRel(WidenBf16(tr_a.x_post_attn[static_cast<size_t>(il)]),
             r4dx_test::LoadNpyF32(FixPath("fixture_a", "x_post_attn_l" + std::to_string(il)),
                                   {B, hidden}),
             ("x_post_attn l" + std::to_string(il)).c_str(), kTolLayerX);
    CheckRel(WidenBf16(tr_a.x_post_ffn[static_cast<size_t>(il)]),
             r4dx_test::LoadNpyF32(FixPath("fixture_a", "x_post_ffn_l" + std::to_string(il)),
                                   {B, hidden}),
             ("x_post_ffn l" + std::to_string(il)).c_str(), kTolLayerX);
  }
  CheckRel(WidenBf16(tr_a.x_final_normed),
           r4dx_test::LoadNpyF32(FixPath("fixture_a", "x_final_normed"), {B, hidden}),
           "x_final_normed", kTolFinal);
  const DflashDraftResult res_a_sel =
      CheckSelector(draft, stream, arena, "fixture_a", anchor_a, 0.0f, B, vocab, embed);
  DflashRoundTrace tr_a_e2e;
  ReportEndToEnd(draft, stream, arena, "fixture_a", anchor_a, 0.0f, B, vocab, embed, lm_head,
                 &tr_a_e2e);

  // ---- Part 2: fixture C (same state, p_min) -----------------------------------------------------
  std::printf("[Part 2] fixture C (p_min=%.2f on fixture A's injected state)\n", p_min_c);
  if (anchor_c != anchor_a) Fail("FAIL: fixture C's anchor differs from A's -- not the same setup\n");
  {
    const DflashDraftResult res_c =
        CheckSelector(draft, stream, arena, "fixture_c", anchor_c, p_min_c, B, vocab, embed);
    std::printf("  %-34s stopped_by_p_min=%d walk_len=%lld (A's walk was %lld)\n", "early stop",
                (int)res_c.stopped_by_p_min, (long long)res_c.walk_len,
                (long long)res_a_sel.walk_len);
    if (!res_c.stopped_by_p_min || res_c.walk_len >= res_a_sel.walk_len) {
      Fail("FAIL: p_min=%.2f did not actually stop the walk earlier than p_min=0 did\n", p_min_c);
    }
    // The n_min discard policy (docs/dflash2.md section 4.3) is a pure host-side gate on the same
    // walk: with n_min one past what the walk produced, the WHOLE draft must be thrown away.
    auto dev = std::make_shared<DeviceBuffer<float>>();
    const DflashLmHeadProvider fixed = FixedLogitsProvider(
        r4dx_test::LoadNpyF32(FixPath("fixture_c", "logits"), {B, vocab}), vocab, dev);
    const DflashDraftResult res_nmin = draft.DraftRound(stream, arena, anchor_c, B - 1, p_min_c,
                                                        res_c.walk_len + 1, embed, fixed, nullptr);
    arena.Reset();
    const bool ok = res_nmin.tokens.empty() && res_nmin.discarded_by_n_min &&
                    res_nmin.walk_len == res_c.walk_len;
    std::printf("  %-34s tokens=%zu discarded=%d walk_len=%lld %s\n", "n_min discard",
                res_nmin.tokens.size(), (int)res_nmin.discarded_by_n_min,
                (long long)res_nmin.walk_len, ok ? "[PASS]" : "[FAIL]");
    if (!ok) ++g_failures;

    // `k` caps the emitted chain. k==0 must draft NOTHING (a driver uses it to fall back to a plain
    // verify of the anchor alone); k==2 must be exactly the k==7 chain's first two tokens, since the
    // walk is a prefix-deterministic greedy chain.
    for (int64_t kcap : {int64_t{0}, int64_t{1}, int64_t{2}}) {
      const DflashDraftResult r =
          draft.DraftRound(stream, arena, anchor_a, kcap, 0.0f, 0, embed,
                           FixedLogitsProvider(r4dx_test::LoadNpyF32(
                                                   FixPath("fixture_a", "logits"), {B, vocab}),
                                               vocab, std::make_shared<DeviceBuffer<float>>()),
                           nullptr);
      arena.Reset();
      bool kok = static_cast<int64_t>(r.tokens.size()) == kcap;
      for (int64_t i = 0; i < kcap && kok; ++i) kok = r.tokens[static_cast<size_t>(i)] ==
                                                       res_a_sel.tokens[static_cast<size_t>(i)];
      std::printf("  %-30s k=%lld  tokens=%zu %s\n", "k cap", (long long)kcap, r.tokens.size(),
                  kok ? "[PASS]" : "[FAIL]");
      if (!kok) ++g_failures;
    }
  }

  // ---- Part 3: fixture B (sliding window + ring wrap) --------------------------------------------
  std::printf("[Part 3] fixture B (N=%lld injected -- 2048 window clips, ring wraps)\n",
              (long long)n_inj_b);
  {
    std::vector<float> feat_b(static_cast<size_t>(n_inj_b * feat_cols));
    r4dx_test::NumpyLegacyRng rng(0);
    rng.FillRandnF32(feat_b.data(), feat_b.size());
    // Same stream, so B's first n_inj_a rows must be bit-identical to A's -- a cheap guard that the
    // generator did not drift over the far larger draw.
    for (int64_t i = 0; i < n_inj_a * feat_cols; ++i) {
      if (feat_b[static_cast<size_t>(i)] != feat_a[static_cast<size_t>(i)]) {
        Fail("FAIL: fixture B's regenerated features diverge from A's at element %lld\n",
             (long long)i);
        break;
      }
    }
    draft.Reset();
    InjectAll(draft, stream, arena, feat_b, n_inj_b, feat_cols);
    if (draft.InjectedCount() != n_inj_b) {
      Fail("FAIL: InjectedCount()=%lld after injecting %lld rows\n",
           (long long)draft.InjectedCount(), (long long)n_inj_b);
    }
    CheckSelector(draft, stream, arena, "fixture_b", anchor_b, 0.0f, B, vocab, embed,
                  kTolGateWideWindow);
    ReportEndToEnd(draft, stream, arena, "fixture_b", anchor_b, 0.0f, B, vocab, embed, lm_head,
                   nullptr);
  }

  // ---- Part 4: w4a16 drift ------------------------------------------------------------------------
  if (!FileExists(kDraftW4a16)) {
    std::printf("[Part 4] SKIPPED: %s not present\n", kDraftW4a16);
  } else {
    std::printf("[Part 4] w4a16 draft container vs bf16, fixture A inputs (drift is expected)\n");
    DflashDraftOptions q = opts;
    q.container_path = kDraftW4a16;
    q.layout = Layout::kW4a16;
    DflashDraft dq = DflashDraft::Load(q);
    InjectAll(dq, stream, arena, feat_a, n_inj_a, feat_cols);
    // Where the 4-bit error enters: the encoder, the injected KV, then layer by layer. Printed as a
    // profile rather than one end number so a future regression can be localised to a stage instead
    // of guessed at (and so a genuine w4a16 PACKING bug, which would show as a step change at one
    // stage, is distinguishable from ordinary quantization noise, which accumulates smoothly).
    std::printf("  drift profile vs the fp32 reference (bf16 container in parentheses):\n");
    std::printf("    g                          %.3e\n",
                RelL2(WidenBf16(dq.DebugEncodedG(stream, n_inj_a)),
                      r4dx_test::LoadNpyF32(FixPath("fixture_a", "g_encoded"), {n_inj_a, hidden})));
    for (int64_t il = 0; il < n_layer; ++il) {
      const std::vector<float> rk = r4dx_test::LoadNpyF32(
          FixPath("fixture_a", "injected_k_l" + std::to_string(il)),
          {n_inj_a, draft.Config().attention.head_count_kv, draft.Config().attention.key_length});
      std::printf("    injected K l%lld              %.3e\n", (long long)il,
                  RelL2(WidenBf16(dq.DebugStoreK(stream, il, 0, n_inj_a)), rk));
    }
    // Same substituted-logits round the bf16 container got, so the ONLY difference between the two
    // traces is the draft weights' own layout -- which is exactly what this part measures.
    auto dev = std::make_shared<DeviceBuffer<float>>();
    const DflashLmHeadProvider fixed = FixedLogitsProvider(
        r4dx_test::LoadNpyF32(FixPath("fixture_a", "logits"), {B, vocab}), vocab, dev);
    DflashRoundTrace tr_q;
    const DflashDraftResult res_q =
        dq.DraftRound(stream, arena, anchor_a, B - 1, 0.0f, 0, embed, fixed, &tr_q);
    arena.Reset();
    for (int64_t il = 0; il < n_layer; ++il) {
      std::printf("    x_post_ffn l%lld              %.3e   (%.3e)\n", (long long)il,
                  RelL2(WidenBf16(tr_q.x_post_ffn[static_cast<size_t>(il)]),
                        r4dx_test::LoadNpyF32(
                            FixPath("fixture_a", "x_post_ffn_l" + std::to_string(il)), {B, hidden})),
                  RelL2(WidenBf16(tr_a.x_post_ffn[static_cast<size_t>(il)]),
                        r4dx_test::LoadNpyF32(
                            FixPath("fixture_a", "x_post_ffn_l" + std::to_string(il)), {B, hidden})));
    }
    std::printf("  with the fixture's logits substituted (isolates the draft weights):\n");
    std::printf("    x_final RelL2 vs reference = %.3e (bf16 container: %.3e)\n",
                RelL2(WidenBf16(tr_q.x_final_normed),
                      r4dx_test::LoadNpyF32(FixPath("fixture_a", "x_final_normed"), {B, hidden})),
                RelL2(WidenBf16(tr_a.x_final_normed),
                      r4dx_test::LoadNpyF32(FixPath("fixture_a", "x_final_normed"), {B, hidden})));
    std::printf("    gate RelL2 vs reference    = %.3e (bf16 container: %.3e)\n",
                RelL2(tr_q.gate, r4dx_test::LoadNpyF32(FixPath("fixture_a", "gate"), {B, 256})),
                RelL2(tr_a.gate, r4dx_test::LoadNpyF32(FixPath("fixture_a", "gate"), {B, 256})));
    size_t agree = 0;
    while (agree < res_q.tokens.size() && agree < res_a_sel.tokens.size() &&
           res_q.tokens[agree] == res_a_sel.tokens[agree]) {
      ++agree;
    }
    std::printf("    chain=[");
    for (size_t i = 0; i < res_q.tokens.size(); ++i) std::printf("%s%d", i ? "," : "", res_q.tokens[i]);
    std::printf("]  agreeing prefix vs bf16 = %zu/%zu\n", agree, res_a_sel.tokens.size());

    // And the full end-to-end path, so the two containers are also compared through a real lm_head.
    DflashRoundTrace tr_q_e2e;
    const DflashDraftResult res_qe =
        dq.DraftRound(stream, arena, anchor_a, B - 1, 0.0f, 0, embed, lm_head, &tr_q_e2e);
    arena.Reset();
    int64_t overlap = 0;
    for (int64_t i = 0; i < B * 16; ++i) {
      const int64_t t = i / 16;
      for (int64_t j2 = 0; j2 < 16; ++j2) {
        if (tr_q_e2e.cand[static_cast<size_t>(i)] == tr_a_e2e.cand[static_cast<size_t>(t * 16 + j2)]) {
          ++overlap;
          break;
        }
      }
    }
    std::printf("  end-to-end vs the bf16 container's own end-to-end run:\n");
    std::printf("    logits RelL2=%.3e  cand set-overlap %lld/%lld (%.1f%%)\n",
                RelL2(tr_q_e2e.logits, tr_a_e2e.logits), (long long)overlap, (long long)(B * 16),
                100.0 * (double)overlap / (double)(B * 16));
    // Reported, not gated: 4-bit draft weights are expected to drift. The only hard requirement is
    // that the round runs, produces in-vocab ids, and does not fall apart entirely.
    for (int32_t id : res_qe.tokens) {
      if (id < 0 || id >= vocab) Fail("FAIL: w4a16 drafted an out-of-vocab id %d\n", id);
    }
    if (res_qe.tokens.empty()) Fail("FAIL: w4a16 drafted nothing at all\n");
  }

  // ---- Part 5: cold-ring gap (docs/dflash2.md section 5, InjectFeatures' relaxed invariant) ------
  // A caller may stop feeding the drafter and resume at a HIGHER absolute position (the server's
  // per-request injection toggle). The rows below the resume point are then stale and must be
  // invisible -- not "nearly invisible", invisible: the round's whole device pipeline runs the
  // identical kernel sequence either way (the visible key count `n_injected - lo` is the same), so
  // the only possible difference is whether those bytes enter the attention scores at all. The gate
  // is therefore BIT-EQUALITY, not a tolerance.
  //
  // Run A: inject rows [0,40), then the SAME 40 rows again at position 100 -- a 60-position gap
  //        whose ring slots hold run A's own first injection (real, plausible, non-zero data, which
  //        is a much harder probe than zeros would be: reading it produces a valid-looking answer).
  // Run B: a ring that has ONLY ever seen rows at [100,140), from Reset().
  // Both then draft at n=140 with the same anchor through the same real synthetic lm_head.
  std::printf("[Part 5] cold-ring gap: rows [0,40) then a resume at 100 must be invisible\n");
  {
    const int64_t kGapResume = 100;
    DflashRoundTrace tr_gap, tr_fresh;
    DflashDraftResult res_gap, res_fresh;

    draft.Reset();
    InjectAll(draft, stream, arena, feat_a, n_inj_a, feat_cols, /*base_pos=*/0);
    if (draft.ValidFrom() != 0) {
      Fail("FAIL: ValidFrom()=%lld after a plain append-only injection (expected 0)\n",
           (long long)draft.ValidFrom());
    }
    InjectAll(draft, stream, arena, feat_a, n_inj_a, feat_cols, /*base_pos=*/kGapResume);
    if (draft.ValidFrom() != kGapResume || draft.InjectedCount() != kGapResume + n_inj_a) {
      Fail("FAIL: after the gap, ValidFrom()=%lld InjectedCount()=%lld (expected %lld / %lld)\n",
           (long long)draft.ValidFrom(), (long long)draft.InjectedCount(), (long long)kGapResume,
           (long long)(kGapResume + n_inj_a));
    }
    res_gap = draft.DraftRound(stream, arena, anchor_a, B - 1, 0.0f, 0, embed, lm_head, &tr_gap);
    arena.Reset();

    draft.Reset();
    InjectAll(draft, stream, arena, feat_a, n_inj_a, feat_cols, /*base_pos=*/kGapResume);
    if (draft.ValidFrom() != kGapResume || draft.InjectedCount() != kGapResume + n_inj_a) {
      Fail("FAIL: fresh-from-Reset injection at %lld gave ValidFrom()=%lld InjectedCount()=%lld\n",
           (long long)kGapResume, (long long)draft.ValidFrom(), (long long)draft.InjectedCount());
    }
    res_fresh =
        draft.DraftRound(stream, arena, anchor_a, B - 1, 0.0f, 0, embed, lm_head, &tr_fresh);
    arena.Reset();

    const bool xf_same = tr_gap.x_final_normed == tr_fresh.x_final_normed;
    const bool logits_same = tr_gap.logits == tr_fresh.logits;
    const bool chain_same = res_gap.tokens == res_fresh.tokens;
    std::printf("  %-34s x_final=%s logits=%s chain=%s %s\n", "gap rows invisible (bit-exact)",
                xf_same ? "eq" : "NE", logits_same ? "eq" : "NE", chain_same ? "eq" : "NE",
                (xf_same && logits_same && chain_same) ? "[PASS]" : "[FAIL]");
    if (!(xf_same && logits_same && chain_same)) {
      ++g_failures;
      std::printf("    the pre-gap rows [0,40) leaked into the round -- store_begin/ValidFrom is "
                  "not reaching the attention kernel\n");
    }

    // The same check one step further: injecting BELOW the frontier must still throw (that is the
    // rollback direction docs/dflash2.md section 5 rules out, and the half of the old append-only
    // check that is deliberately kept).
    bool threw = false;
    try {
      DeviceBuffer<uint16_t> stage(static_cast<size_t>(feat_cols));
      stage.Zero();
      draft.InjectFeatures(stream, arena, stage.data(), 1, draft.InjectedCount() - 1);
    } catch (const std::exception&) {
      threw = true;
    }
    arena.Reset();
    std::printf("  %-34s threw=%s %s\n", "inject below frontier rejected", threw ? "yes" : "NO",
                threw ? "[PASS]" : "[FAIL]");
    if (!threw) ++g_failures;
  }

  std::printf(g_failures == 0 ? "[PASS] test_dflash_draft: all checks passed\n"
                              : "[FAIL] test_dflash_draft: %d check(s) failed\n",
              g_failures);
  return g_failures == 0 ? 0 : 1;
}
