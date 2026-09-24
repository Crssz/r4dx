// tests/model/test_teacher_forced_logprobs.cpp -- the automatic regression for the Rung 4 dump
// pass (tests/model/teacher_forced.h, driven by tool_teacher_forced_logprobs on the real
// container). Runs the 4-layer self-consistency check that the tool's own `--check-greedy` runs,
// so the pass's two invariants are enforced by ctest on every build instead of only by whoever
// remembers to run the tool by hand:
//
//   1. GREEDY CONSISTENCY. Generate N tokens greedily from a fixed prompt, then teacher-force the
//      WHOLE resulting sequence (prompt + generated) through the dump pass. For every row that
//      predicts a generated token, the row's argmax must be exactly that token. This is the check
//      that would catch the dump being off by one position, reading the wrong row, or silently
//      feeding the sequence into a different state than generation did -- none of which a
//      "looks like a probability distribution" check can see.
//   2. NORMALIZATION. |log sum_j exp(row[j])| < 1e-2 for every row, i.e. the dumped rows really are
//      log-probabilities.
//
// The one row the first check deliberately does NOT cover is the row that predicts the FIRST
// generated token: generation computed it with the chunked-PREFILL GDN kernels (one Model::Prefill
// call over the whole prompt) while the dump computes it with the per-token decode kernels, so the
// two agree only to GPU reduction-order noise, not bit for bit, and an argmax equality assertion
// there would be a flaky test rather than a contract. Every row from the second generated token on
// is computed by the identical call in both passes and IS asserted. (kPromptTokens' last row is
// therefore skipped by starting the check one row later -- see kGeneratedTokens/check_greedy_last_n
// below.)
//
// Needs the 4-layer test container (D:/models/r4dx/qwen38-27b-l4-mtp.r4dx, bf16 + w4a16) and HIP
// device 1; exits 77 (CTest SKIPPED, not FAILED) when the container is absent, the convention every
// real-container test in this directory uses. Deliberately uses hand-picked token IDS rather than a
// tokenizer, so it links nothing beyond what the other tests/model targets already do.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "local_text_model.h"
#include "model.h"
#include "teacher_forced.h"
#include "test_common.h"

using r4dx_test::FileExists;
using r4dx_test::SkipMissing;
using r4dx::model::Model;
using r4dx::model::ModelOptions;

namespace {

const char* kContainerPath = r4dx_test::ContainerPath("D:/models/r4dx/qwen38-27b-l4-mtp.r4dx");
constexpr int64_t kLayers = 4;      // the container is a --layers 4 truncation; its config.json
                                     // still declares the full 64 (see test_forward_smoke.cpp)
constexpr int64_t kGeneratedTokens = 64;

// A short fixed prompt. Any in-vocabulary ids work: this test asserts SELF-consistency (the dump
// reproduces what this same container greedily generated), not that the text means anything -- a
// 4-layer truncation of a 27B model produces nothing meaningful either way.
const std::vector<int32_t>& PromptTokens() {
  static const std::vector<int32_t> kPrompt = {785, 3974, 13876, 38835, 34208, 916, 279, 15678};
  return kPrompt;
}

}  // namespace

int main() {
  if (!FileExists(kContainerPath)) return SkipMissing(kContainerPath);
  try {
    ModelOptions opts;
    opts.container_path = kContainerPath;
    opts.layout = r4dx::model::LayoutFromName("w4a16");
    opts.max_ctx = 512;
    opts.layer_limit = kLayers;
    opts.vision = ModelOptions::VisionMode::kOff;
    // Through the TP=1 TextModel, exactly as tool_teacher_forced_logprobs drives it (docs/tp.md 2.8):
    // every call below forwards one-to-one to the Model.
    r4dx::model::LocalTextModel model(Model::Load(opts));

    // ---- 1. greedy generation, exactly as r4dx-cli --temperature 0 --mtp 0 does it --------------
    // DecodeStep (full logits) rather than DecodeStepGreedy, purely so this loop can record the
    // same per-row fingerprint RunSegment reports and check 3 below can compare them. The emitted
    // token is identical either way (DecodeStepGreedy is the same step with the argmax done on the
    // device), so this is still the CLI's own greedy sequence.
    const std::vector<int32_t>& prompt = PromptTokens();
    std::vector<int32_t> seq = prompt;
    std::vector<int32_t> gen_argmax;  // per generated row, in row order
    std::vector<double> gen_lse;
    auto fingerprint = [&](const std::vector<float>& logits) {
      const int64_t am = std::max_element(logits.begin(), logits.end()) - logits.begin();
      double sum = 0.0;
      const double m = logits[static_cast<size_t>(am)];
      for (float v : logits) sum += std::exp(static_cast<double>(v) - m);
      gen_argmax.push_back(static_cast<int32_t>(am));
      gen_lse.push_back(static_cast<double>(static_cast<float>(m + std::log(sum))));
      return static_cast<int32_t>(am);
    };
    const std::vector<float> prompt_logits = model.Prefill(prompt);
    // Row prompt.size()-1 is the prefill-path row this test deliberately excludes (see the header
    // comment), so its fingerprint is not recorded.
    int32_t next = static_cast<int32_t>(
        std::max_element(prompt_logits.begin(), prompt_logits.end()) - prompt_logits.begin());
    for (int64_t i = 0; i < kGeneratedTokens; ++i) {
      seq.push_back(next);
      if (i + 1 < kGeneratedTokens) next = fingerprint(model.DecodeStep(next));
    }
    std::printf("[gen] %zu prompt + %lld generated = %zu tokens; first generated=%d last=%d\n",
                prompt.size(), static_cast<long long>(kGeneratedTokens), seq.size(), seq[prompt.size()],
                seq.back());

    // ---- 2. teacher-force the whole sequence through the Rung 4 pass ---------------------------
    r4dx_tf::Segment seg;
    seg.name = "l4_selfcheck";
    seg.token_ids = seq;
    r4dx_tf::SegmentOptions so;
    so.out_dir.clear();  // compute everything, write nothing -- a ctest leaves no artifacts behind
    so.layout_name = "w4a16";
    // rows are indexed 0..T-2 and row i predicts token i+1, so the rows predicting a GENERATED
    // token are the last kGeneratedTokens of them. Checking one fewer skips exactly the
    // prefill-path-vs-decode-path row described in this file's header comment.
    so.check_greedy_last_n = kGeneratedTokens - 1;
    so.progress_every = 0;
    so.container_path = kContainerPath;
    const r4dx_tf::SegmentResult r = r4dx_tf::RunSegment(model, seg, so);

    std::printf("[dump] T=%lld rows=%lld vocab=%lld in %.2fs (%.1f ms/row)\n",
                static_cast<long long>(r.T), static_cast<long long>(r.rows),
                static_cast<long long>(r.V), r.wall_s,
                1000.0 * r.wall_s / static_cast<double>(std::max<int64_t>(1, r.rows)));
    std::printf("[dump] sha256(token_ids json) = %s\n", r.sha256.c_str());

    int failures = 0;
    std::printf("[check] greedy consistency: %lld/%lld rows argmax == next token\n",
                static_cast<long long>(r.greedy_checked - r.greedy_mismatches),
                static_cast<long long>(r.greedy_checked));
    if (r.greedy_checked != kGeneratedTokens - 1) {
      std::fprintf(stderr, "FAIL: expected %lld checked rows, got %lld\n",
                   static_cast<long long>(kGeneratedTokens - 1),
                   static_cast<long long>(r.greedy_checked));
      ++failures;
    }
    if (r.greedy_mismatches != 0) {
      std::fprintf(stderr,
                   "FAIL: %lld greedy mismatch(es); first at row %lld: argmax=%d, token_ids[%lld]=%d\n",
                   static_cast<long long>(r.greedy_mismatches),
                   static_cast<long long>(r.first_mismatch_row), r.first_mismatch_got,
                   static_cast<long long>(r.first_mismatch_row + 1), r.first_mismatch_want);
      ++failures;
    }
    std::printf("[check] normalization: max |logsumexp(row)| = %.3e (tol 1e-2)\n", r.max_abs_lse);
    if (!(r.max_abs_lse < 1e-2)) {
      std::fprintf(stderr, "FAIL: rows are not normalized log-probabilities\n");
      ++failures;
    }

    // ---- 3. row alignment against the generation loop's own logits -----------------------------
    // The greedy check above is only as strong as the generated tail is varied, and a 4-layer
    // truncation of a 27B model routinely collapses onto one repeated token -- against which an
    // off-by-one row would still "pass". This check does not care: for every row the dump shares
    // with the generation loop it compares the row's logsumexp, a single scalar that depends on all
    // V raw logits, so a dump reading row i-1 or i+1 fails it even when every argmax is identical.
    //
    // The bound is a SEPARATION bound, not a modelling tolerance, and the test computes both sides
    // of the separation rather than asserting a magic number: `spread` is how far apart NEIGHBOURING
    // rows' logsumexps are (what an off-by-one would cost), `worst_lse_delta` is how far apart the
    // two passes' SAME row is. The two passes are not bit-identical even though both rows come out
    // of Model::DecodeStep at the same position: the generation loop fed the prompt through ONE
    // Model::Prefill call (the chunked-scan GDN kernels) while the dump feeds it one token at a
    // time (the recurrent decode kernels), so every shared row inherits that prompt-state
    // difference. Measured on this fixture: ~7e-4 against a neighbour spread of order 1.
    const int64_t first_shared_row = static_cast<int64_t>(prompt.size());  // row p, predicting gen[1]
    double worst_lse_delta = 0.0, spread = 0.0;
    int64_t compared = 0, distinct_lse = 0;
    double prev_lse = 0.0;
    for (size_t k = 0; k < gen_lse.size(); ++k) {
      const size_t row = static_cast<size_t>(first_shared_row) + k;
      if (row >= r.lse.size()) break;
      worst_lse_delta = std::max(worst_lse_delta, std::fabs(r.lse[row] - gen_lse[k]));
      if (r.argmax[row] != gen_argmax[k]) {
        std::fprintf(stderr, "FAIL: row %zu argmax %d from the dump vs %d from generation\n", row,
                     r.argmax[row], gen_argmax[k]);
        ++failures;
      }
      if (compared == 0 || r.lse[row] != prev_lse) ++distinct_lse;
      if (compared > 0) spread = std::max(spread, std::fabs(r.lse[row] - prev_lse));
      prev_lse = r.lse[row];
      ++compared;
    }
    std::printf("[check] row alignment: %lld shared rows, max |logsumexp delta| = %.3e, "
                "neighbour spread = %.3e, %lld distinct row logsumexp value(s)\n",
                static_cast<long long>(compared), worst_lse_delta, spread,
                static_cast<long long>(distinct_lse));
    if (compared != kGeneratedTokens - 1) {
      std::fprintf(stderr, "FAIL: expected %lld shared rows, compared %lld\n",
                   static_cast<long long>(kGeneratedTokens - 1), static_cast<long long>(compared));
      ++failures;
    }
    if (!(worst_lse_delta < 1e-2)) {
      std::fprintf(stderr, "FAIL: the dump's rows do not line up with the generation loop's own "
                            "logits (max |logsumexp delta| = %.3e)\n", worst_lse_delta);
      ++failures;
    }
    // Guard against the check silently becoming vacuous: if this fixture ever degenerates to the
    // point where neighbouring rows are indistinguishable too, an off-by-one really would slip
    // through and this test would need a different prompt/container, not a quiet pass. Require an
    // order of magnitude of headroom between "same row, other pass" and "next row".
    if (distinct_lse < 2 || !(spread > 10.0 * worst_lse_delta)) {
      std::fprintf(stderr, "FAIL: neighbouring rows' logsumexp spread (%.3e) does not separate from "
                            "the two passes' own disagreement (%.3e) -- the row-alignment check is "
                            "vacuous on this fixture; pick a different prompt\n", spread,
                   worst_lse_delta);
      ++failures;
    }

    // The provenance hash the shared format carries must be stable and well-formed -- the KL report
    // pairs the two halves on it, so a silently empty/short digest would defeat the pairing.
    if (r.sha256.size() != 64) {
      std::fprintf(stderr, "FAIL: sha256_of_token_ids_json is %zu chars, expected 64\n",
                   r.sha256.size());
      ++failures;
    }
    if (r4dx_tf::TokenIdsSha256(seq) != r.sha256) {
      std::fprintf(stderr, "FAIL: TokenIdsSha256 is not deterministic\n");
      ++failures;
    }
    // Known-answer test for the SHA-256 implementation itself (FIPS 180-4 "abc").
    {
      r4dx_tf::Sha256 h;
      h.Update(std::string("abc"));
      const std::string got = h.HexDigest();
      const std::string want = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
      if (got != want) {
        std::fprintf(stderr, "FAIL: sha256(\"abc\") = %s, expected %s\n", got.c_str(), want.c_str());
        ++failures;
      }
    }

    if (failures != 0) {
      std::fprintf(stderr, "test_teacher_forced_logprobs: %d check(s) failed\n", failures);
      return 1;
    }
    std::printf("[PASS] teacher-forced log-prob pass is self-consistent on the 4-layer container\n");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "test_teacher_forced_logprobs: %s\n", e.what());
    return 1;
  }
  return 0;
}
