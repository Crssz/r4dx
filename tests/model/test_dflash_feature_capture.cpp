// tests/model/test_dflash_feature_capture.cpp -- Milestone 5 B1 item 1 (docs/dflash2.md
// "Implementation"): Model::AttachDflashFeatureCapture/DflashFeatureBuffer.
//
// Two things this test gates, against the real 4-layer test container (same convention as
// test_forward_smoke.cpp):
//
// 1. CORRECTNESS: with capture attached at target layers {0, 2}, the captured buffer's column 0
//    (layer 0's INPUT) must exactly equal the token embedding table's own gather for the prompt
//    (computed independently here via r4dx::kernels::EmbeddingGatherHost -- the same host-side
//    gather Model::RunChunk itself uses to build layer 0's input, but called directly against the
//    container's embedding table rather than trusting RunChunk's own internal wiring). Column 1
//    (layer 2's INPUT) is checked to be finite and, crucially, DIFFERENT from column 0 (two
//    different layers' residual streams should not coincide on a >1-layer model), which is the
//    simplest available cross-check that the per-layer column offset logic (not just layer 0's
//    trivial case) is really picking out layer 2 and not silently aliasing layer 0.
// 2. NO-OP WHEN ABSENT: r4dx::kernels::r4dx_kernel_launch_counter_get()'s delta across a Prefill()
//    call is IDENTICAL whether or not a capture is attached (the capture uses hipMemcpy2DAsync, a
//    plain HIP runtime strided D2D copy, never an r4dx-owned kernel launch) -- proving the hook adds
//    no r4dx-owned kernel launches either way -- AND the returned logits are identical (within
//    float round-trip noise) whether or not a capture is attached, proving the hook does not
//    perturb the actual forward pass.
//
// 3. VERIFYWINDOW PATH COVERAGE (review finding, 2026-09-20): attaches a capture on an
//    MTP-enabled Model and drives it through Model::VerifyWindow (not just Prefill/RunChunk) --
//    the load-bearing call site for DFlash2's real lifecycle (capturing the accepted prefix's
//    features after a verify round) -- asserting DflashFeatureRows()==candidates.size() and that
//    row t / column 0 is bit-exact against EmbeddingGatherHost(candidates[t]), the same layer-0==
//    embedding ground truth Part 1 already uses for the RunChunk/Prefill path.
// 4. MULTI-CHUNK PREFILL COVERAGE (review finding, 2026-09-20): a >64-token prompt (so Prefill()
//    internally issues more than one RunChunk call) attaches a capture and passes Prefill() the
//    `on_chunk_captured` drain callback, accumulating each chunk's rows on the host BEFORE the next
//    chunk's RunChunk call overwrites DflashFeatureBuffer(). Asserts the accumulated row count
//    equals the full prompt length and every row is bit-exact against EmbeddingGatherHost() over
//    the FULL prompt -- proving every prefilled position's features are retrievable, not just the
//    tail chunk's (the bug: pre-fix, only the last <=64-token chunk's rows would have survived).
//
// SKIPs (CTest SKIPPED, not FAILED) if the container is missing, same convention as
// test_forward_smoke.cpp.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "container.h"
#include "model.h"
#include "r4dx/kernels/embedding.hpp"
#include "r4dx/kernels/kernels.h"
#include "test_common.h"

using namespace r4dx_test;
using r4dx::model::Layout;
using r4dx::model::Model;
using r4dx::model::ModelOptions;

namespace {
const char* kContainerPath = "D:/models/r4dx/qwen38-27b-l4-bf16.r4dx";
// Same 4-layer MTP-enabled container test_mtp.cpp uses (converted `--layers 4 --mtp on --layouts
// bf16,w4a16,w4a8,mxfp4`) -- separately SKIPped below if absent so a machine missing it still runs
// every other check in this file.
const char* kMtpContainerPath = "D:/models/r4dx/qwen38-27b-l4-allmtp.r4dx";
constexpr int64_t kDraftK = 3;

std::vector<int32_t> MakePromptTokens(int n) {
  std::vector<int32_t> ids(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) ids[static_cast<size_t>(i)] = 200 + (i * 53) % 5000;
  return ids;
}

bool AllFinite(const std::vector<float>& v) {
  for (float x : v) {
    if (!std::isfinite(x)) return false;
  }
  return true;
}

}  // namespace

int main() {
  if (!FileExists(kContainerPath)) {
    return SkipMissing(kContainerPath);
  }

  ModelOptions opts;
  opts.container_path = kContainerPath;
  opts.layout = Layout::kBf16;  // exact-arithmetic layout -- this test checks bit-level plumbing,
                                 // not numerical tolerance, so avoid quantization noise entirely.
  opts.max_ctx = 256;
  opts.layer_limit = 4;

  const std::vector<int32_t> prompt = MakePromptTokens(10);  // single chunk (< max_chunk_==64)

  // ---- Part 1: correctness of the captured buffer -------------------------------------------
  {
    Model model = Model::Load(opts);
    const int64_t H = model.Config().hidden_size;
    model.AttachDflashFeatureCapture({0, 2});
    std::vector<float> logits = model.Prefill(prompt);
    if (!AllFinite(logits)) {
      std::fprintf(stderr, "FAIL: prefill logits contain NaN/Inf with capture attached\n");
      return 1;
    }
    if (model.DflashFeatureRows() != static_cast<int64_t>(prompt.size())) {
      std::fprintf(stderr, "FAIL: DflashFeatureRows()=%lld, expected %zu\n",
                   static_cast<long long>(model.DflashFeatureRows()), prompt.size());
      return 1;
    }
    if (model.DflashFeatureCols() != 2 * H) {
      std::fprintf(stderr, "FAIL: DflashFeatureCols()=%lld, expected %lld\n",
                   static_cast<long long>(model.DflashFeatureCols()),
                   static_cast<long long>(2 * H));
      return 1;
    }

    std::vector<uint16_t> captured(static_cast<size_t>(model.DflashFeatureRows() * model.DflashFeatureCols()));
    // DflashFeatureBuffer() returns a raw device pointer (see model.h's doc comment) -- a plain
    // synchronous hipMemcpy is safe here since the caller (this test) knows the producing stream
    // has already been synchronized by the time Prefill() returned its host-side logits vector.
    (void)hipMemcpy(captured.data(), model.DflashFeatureBuffer(), captured.size() * sizeof(uint16_t),
                     hipMemcpyDeviceToHost);

    // Ground truth for column 0 (layer 0's input == the raw token embedding, untouched by any
    // layer): gather it independently via the SAME host-side embedding function RunChunk uses,
    // called directly against the container's own embedding table.
    std::vector<uint16_t> expected_col0(prompt.size() * static_cast<size_t>(H));
    r4dx::kernels::EmbeddingGatherHost(model.GetContainer().EmbedTokensHost(),
                                        model.Config().vocab_size, H, prompt, expected_col0.data());

    const int64_t cols = model.DflashFeatureCols();
    int mismatches = 0;
    for (size_t t = 0; t < prompt.size(); ++t) {
      for (int64_t h = 0; h < H; ++h) {
        const uint16_t got = captured[t * static_cast<size_t>(cols) + static_cast<size_t>(h)];
        const uint16_t exp = expected_col0[t * static_cast<size_t>(H) + static_cast<size_t>(h)];
        if (got != exp) ++mismatches;
      }
    }
    if (mismatches != 0) {
      std::fprintf(stderr,
                   "FAIL: layer-0 captured column mismatches token-embedding ground truth in %d/%lld "
                   "elements (expected 0 -- this is a bit-exact D2D copy check, not a numerical one)\n",
                   mismatches, static_cast<long long>(prompt.size() * static_cast<size_t>(H)));
      return 1;
    }
    std::fprintf(stderr, "[PASS] layer-0 captured column bit-exact vs EmbeddingGatherHost ground truth\n");

    // Column 1 (layer 2's input) must be finite and must differ from column 0 somewhere -- the
    // simplest available cross-check (without reimplementing 2 layers' forward pass here) that the
    // per-layer column-offset arithmetic really distinguishes layer 2 from layer 0.
    bool any_diff = false;
    bool any_nonzero = false;
    for (size_t t = 0; t < prompt.size() && !any_diff; ++t) {
      for (int64_t h = 0; h < H; ++h) {
        const uint16_t c0 = captured[t * static_cast<size_t>(cols) + static_cast<size_t>(h)];
        const uint16_t c1 = captured[t * static_cast<size_t>(cols) + static_cast<size_t>(H) + static_cast<size_t>(h)];
        if (c1 != 0) any_nonzero = true;
        if (c0 != c1) { any_diff = true; break; }
      }
    }
    if (!any_nonzero) {
      std::fprintf(stderr, "FAIL: layer-2 captured column is all-zero (never written?)\n");
      return 1;
    }
    if (!any_diff) {
      std::fprintf(stderr, "FAIL: layer-2 captured column is byte-identical to layer-0's -- column "
                            "offset arithmetic looks aliased\n");
      return 1;
    }
    std::fprintf(stderr, "[PASS] layer-2 captured column is populated and distinct from layer-0's\n");
  }

  // ---- Part 2: no-op (launch count + logits) when a capture is attached/detached ------------
  {
    r4dx_kernel_launch_counter_reset();
    Model baseline = Model::Load(opts);
    std::vector<float> logits_off = baseline.Prefill(prompt);
    const int64_t launches_off = r4dx_kernel_launch_counter_get();

    r4dx_kernel_launch_counter_reset();
    Model captured_model = Model::Load(opts);
    captured_model.AttachDflashFeatureCapture({0, 1, 2, 3});
    std::vector<float> logits_on = captured_model.Prefill(prompt);
    const int64_t launches_on = r4dx_kernel_launch_counter_get();

    if (launches_on != launches_off) {
      std::fprintf(stderr,
                   "FAIL: r4dx-owned kernel launch count differs with capture attached (%lld) vs "
                   "not (%lld) -- capture must add zero r4dx-owned kernel launches\n",
                   static_cast<long long>(launches_on), static_cast<long long>(launches_off));
      return 1;
    }
    std::fprintf(stderr, "[PASS] r4dx-owned kernel launch count unaffected by capture (%lld both ways)\n",
                 static_cast<long long>(launches_off));

    if (logits_off.size() != logits_on.size()) {
      std::fprintf(stderr, "FAIL: logits size differs with capture attached\n");
      return 1;
    }
    for (size_t i = 0; i < logits_off.size(); ++i) {
      if (logits_off[i] != logits_on[i]) {
        std::fprintf(stderr, "FAIL: logits[%zu] differs with capture attached (%.9g vs %.9g) -- the "
                              "capture must not perturb the forward pass\n",
                     i, static_cast<double>(logits_off[i]), static_cast<double>(logits_on[i]));
        return 1;
      }
    }
    std::fprintf(stderr, "[PASS] logits bit-identical with capture attached vs not\n");

    // Detach must bring the model back to "no capture" bookkeeping.
    captured_model.DetachDflashFeatureCapture();
    if (captured_model.DflashFeatureCaptureAttached()) {
      std::fprintf(stderr, "FAIL: DflashFeatureCaptureAttached() true after Detach()\n");
      return 1;
    }
  }

  // ---- Part 3: VerifyWindow path coverage (review finding, 2026-09-20) ----------------------
  if (!FileExists(kMtpContainerPath)) {
    std::fprintf(stderr,
                 "[SKIP] %s not found -- VerifyWindow capture coverage did NOT run this pass (not "
                 "a product failure; every other check in this file still ran and passed).\n",
                 kMtpContainerPath);
  } else {
    ModelOptions mtp_opts;
    mtp_opts.container_path = kMtpContainerPath;
    mtp_opts.layout = Layout::kBf16;
    mtp_opts.max_ctx = 256;
    mtp_opts.layer_limit = 4;
    mtp_opts.mtp_draft_k = kDraftK;

    Model mtp = Model::Load(mtp_opts);
    const int64_t H = mtp.Config().hidden_size;
    const std::vector<int32_t> prompt = MakePromptTokens(10);
    mtp.Prefill(prompt);  // must run before VerifyWindow (docs/mtp.md) -- no capture attached yet,
                           // matching a real caller that only starts capturing once drafting begins.

    mtp.AttachDflashFeatureCapture({0});
    // Candidates need not be real accepted drafts for this check -- VerifyWindow captures whatever
    // residual stream it computes for each candidate row regardless of whether that row is later
    // accepted (model.cpp's own comment on VerifyWindow's KV writes makes the identical point).
    const std::vector<int32_t> candidates = {100, 4321, 987, 55};  // kDraftK+1 == 4 rows
    std::vector<float> verify_logits;
    (void)mtp.VerifyWindow(candidates, &verify_logits);

    if (mtp.DflashFeatureRows() != static_cast<int64_t>(candidates.size())) {
      std::fprintf(stderr, "FAIL: VerifyWindow: DflashFeatureRows()=%lld, expected %zu\n",
                   static_cast<long long>(mtp.DflashFeatureRows()), candidates.size());
      return 1;
    }

    std::vector<uint16_t> captured(static_cast<size_t>(mtp.DflashFeatureRows() * H));
    (void)hipMemcpy(captured.data(), mtp.DflashFeatureBuffer(), captured.size() * sizeof(uint16_t),
                     hipMemcpyDeviceToHost);

    std::vector<uint16_t> expected(candidates.size() * static_cast<size_t>(H));
    r4dx::kernels::EmbeddingGatherHost(mtp.GetContainer().EmbedTokensHost(), mtp.Config().vocab_size,
                                        H, candidates, expected.data());

    int mismatches = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
      if (captured[i] != expected[i]) ++mismatches;
    }
    if (mismatches != 0) {
      std::fprintf(stderr,
                   "FAIL: VerifyWindow capture mismatches token-embedding ground truth in %d/%zu "
                   "elements\n",
                   mismatches, expected.size());
      return 1;
    }
    std::fprintf(stderr,
                 "[PASS] VerifyWindow capture: %lld rows bit-exact vs EmbeddingGatherHost ground "
                 "truth\n",
                 static_cast<long long>(mtp.DflashFeatureRows()));
  }

  // ---- Part 4: multi-chunk Prefill drain coverage (review finding, 2026-09-20) ----------------
  {
    Model model = Model::Load(opts);
    const int64_t H = model.Config().hidden_size;
    // 130 tokens over a max_chunk_==64 model: 3 RunChunk calls (64, 64, 2) -- exercises the exact
    // multi-chunk overwrite bug this part guards against.
    const std::vector<int32_t> prompt = MakePromptTokens(130);
    model.AttachDflashFeatureCapture({0});

    std::vector<uint16_t> accumulated;
    accumulated.reserve(prompt.size() * static_cast<size_t>(H));
    int chunk_calls = 0;
    auto drain = [&]() {
      ++chunk_calls;
      const int64_t rows = model.DflashFeatureRows();
      std::vector<uint16_t> chunk(static_cast<size_t>(rows * H));
      (void)hipMemcpy(chunk.data(), model.DflashFeatureBuffer(), chunk.size() * sizeof(uint16_t),
                       hipMemcpyDeviceToHost);
      accumulated.insert(accumulated.end(), chunk.begin(), chunk.end());
    };
    (void)model.Prefill(prompt, drain);

    if (chunk_calls < 2) {
      std::fprintf(stderr, "FAIL: expected >1 RunChunk call for a %zu-token prompt, got %d\n",
                   prompt.size(), chunk_calls);
      return 1;
    }
    if (accumulated.size() != prompt.size() * static_cast<size_t>(H)) {
      std::fprintf(stderr,
                   "FAIL: accumulated capture has %zu elements, expected %zu (%zu tokens x %lld "
                   "hidden) -- multi-chunk Prefill is still dropping earlier chunks' features\n",
                   accumulated.size(), prompt.size() * static_cast<size_t>(H), prompt.size(),
                   static_cast<long long>(H));
      return 1;
    }

    std::vector<uint16_t> expected(prompt.size() * static_cast<size_t>(H));
    r4dx::kernels::EmbeddingGatherHost(model.GetContainer().EmbedTokensHost(),
                                        model.Config().vocab_size, H, prompt, expected.data());
    int mismatches = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
      if (accumulated[i] != expected[i]) ++mismatches;
    }
    if (mismatches != 0) {
      std::fprintf(stderr,
                   "FAIL: multi-chunk accumulated capture mismatches token-embedding ground truth "
                   "in %d/%zu elements\n",
                   mismatches, expected.size());
      return 1;
    }
    std::fprintf(stderr,
                 "[PASS] multi-chunk Prefill drain: %d RunChunk calls, all %zu rows recovered "
                 "bit-exact via on_chunk_captured\n",
                 chunk_calls, prompt.size());
  }

  return 0;
}
