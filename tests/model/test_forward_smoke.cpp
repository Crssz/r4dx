// tests/model/test_forward_smoke.cpp -- end-to-end smoke test for r4dx::model::Model (the full
// assembled forward pass: embedding -> N decoder layers (GDN + full attention, chunked prefill +
// single-token decode) -> final_norm -> lm_head), against the real 4-layer test container
// (layers 0-2 GDN, layer 3 full attention -- the real model's [linear,linear,linear,full] repeat
// pattern -- so this exercises both layer kinds and the KV/GDN state plumbing between them).
//
// Not a golden-accuracy test (tests/model/test_gdn_layer.cpp, test_final_lm_head.cpp and
// tests/model/attention/test_attn_layer.cpp already cover per-component numerical correctness
// against real transformers goldens) -- this only checks that Model::Load/Prefill/DecodeStep run
// to completion and produce finite (no NaN/Inf), correctly-shaped logits, for every layout the
// test container carries, chaining a >64-token prefill (so it must chunk) with several decode
// steps that continue the carried GDN/KV state.
//
// It IS, however, value-gated on one specific invariant: Prefill(tokens[0:N]) and
// Prefill(tokens[0:N-1]) + DecodeStep(tokens[N-1]) both compute "logits for the token that follows
// tokens[N-1]" and must agree. Chunked prefill's chunk boundary (has_init false->true) and the
// prefill->decode kernel switch (chunked-scan vs conv_update+recurrent_update for GDN,
// attn_prefill_fp8kv vs attn_decode_fp8kv for attention) are exactly the state handoffs the
// blocker this stage fixed (GDN conv-state depth) lived in, and neither is exercised by the
// NaN/Inf-only checks below on their own -- a wrong layer ordering, residual/norm placement, KV
// start_pos bookkeeping, or (as the blocker was) an off-by-one conv-state depth would make these
// two paths disagree well outside kernel-level numerical noise while every other check here still
// passes.
//
// SKIPs (CTest SKIPPED, not FAILED) if the container is missing, same convention as
// tests/model/test_gdn_layer.cpp / tests/tokenizer's golden test.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "model.h"
#include "test_common.h"

using namespace r4dx_test;
using r4dx::model::Layout;
using r4dx::model::LayoutName;
using r4dx::model::Model;
using r4dx::model::ModelOptions;

namespace {
const char* kContainerPath = r4dx_test::ContainerPath("D:/models/r4dx/qwen38-27b-l4-bf16.r4dx");

bool AllFinite(const std::vector<float>& v) {
  for (float x : v) {
    if (!std::isfinite(x)) return false;
  }
  return true;
}

// Deterministic, in-vocab (vocab=248320) filler token ids -- this test only checks the pipeline
// runs and stays finite, not that the ids are linguistically meaningful.
std::vector<int32_t> MakePromptTokens(int n) {
  std::vector<int32_t> ids(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) ids[static_cast<size_t>(i)] = 100 + (i * 37) % 5000;
  return ids;
}

double RelL2(const std::vector<float>& got, const std::vector<float>& ref) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double d = static_cast<double>(got[i]) - static_cast<double>(ref[i]);
    num += d * d;
    den += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
  }
  return std::sqrt(num) / std::max(1e-9, std::sqrt(den));
}

// Prefill(tokens) vs Prefill(tokens minus its last token) + DecodeStep(tokens.back()) both compute
// logits for the token that follows tokens.back() and must agree -- see file header comment. Each
// path gets its own freshly-loaded Model so the two runs cannot share any state.
double PrefillDecodeEquivRelErr(const ModelOptions& opts, const std::vector<int32_t>& tokens) {
  Model model_a = Model::Load(opts);
  std::vector<float> logits_a = model_a.Prefill(tokens);

  Model model_b = Model::Load(opts);
  const std::vector<int32_t> prefix(tokens.begin(), tokens.end() - 1);
  model_b.Prefill(prefix);
  std::vector<float> logits_b = model_b.DecodeStep(tokens.back());

  return RelL2(logits_b, logits_a);
}

// Model::Reset() (src/server's cheap alternative to a full Model::Load() on a prefix mismatch,
// docs/server.md "Reset cost") must be indistinguishable from a fresh Load() for everything this
// test exercises: `reused` already carries state from an earlier prefill+decode run (unlike
// PrefillDecodeEquivRelErr's two always-fresh Models above) -- Reset() it in place, replay the
// exact same prompt+decode sequence, and diff against a genuinely fresh Model doing the same
// thing. This is a much tighter bar than the "small measured kernel-noise error" the prefill/
// decode equivalence check above tolerates: Reset()+replay and a fresh Load()+the same calls run
// the IDENTICAL sequence of kernels over IDENTICAL state, so any gap here means Reset() left some
// piece of state (GDN recurrent/conv, MTP seed/accept bookkeeping, pos_/started_) not truly
// reset to its post-Load() value, not ordinary GPU reduction-order noise.
double ResetMatchesFreshLoadRelErr(Model& reused, const ModelOptions& opts,
                                    const std::vector<int32_t>& prompt, int32_t decode_tok) {
  reused.Reset();
  if (reused.PositionCount() != 0) {
    std::fprintf(stderr, "FAIL: Model::Reset() left PositionCount()=%lld, expected 0\n",
                 static_cast<long long>(reused.PositionCount()));
    return 1e9;  // fail the caller's tolerance check unconditionally
  }
  std::vector<float> logits_reset = reused.Prefill(prompt);
  for (int step = 0; step < 5; ++step) logits_reset = reused.DecodeStep(decode_tok);

  Model fresh = Model::Load(opts);
  std::vector<float> logits_fresh = fresh.Prefill(prompt);
  for (int step = 0; step < 5; ++step) logits_fresh = fresh.DecodeStep(decode_tok);

  return RelL2(logits_reset, logits_fresh);
}

}  // namespace

int main() {
  if (!FileExists(kContainerPath)) {
    return SkipMissing(kContainerPath);
  }

  const Layout layouts[] = {Layout::kBf16, Layout::kMxfp4, Layout::kW4a16, Layout::kW4a8};
  int ran = 0;
  for (Layout layout : layouts) {
    ModelOptions opts;
    opts.container_path = kContainerPath;
    opts.layout = layout;
    opts.max_ctx = 256;
    opts.layer_limit = 4;

    Model model = Model::Load(opts);

    // Prefill 80 tokens: forces Model::Prefill to chunk internally (max_chunk_=64), exercising the
    // has_init=false-then-true GDN transition across a chunk boundary and the attention layer's
    // start_pos bookkeeping.
    const std::vector<int32_t> prompt = MakePromptTokens(80);
    std::vector<float> logits = model.Prefill(prompt);
    if (logits.size() != static_cast<size_t>(model.Config().vocab_size)) {
      std::fprintf(stderr, "FAIL [%s]: prefill logits size %zu != vocab %lld\n",
                   LayoutName(layout), logits.size(),
                   static_cast<long long>(model.Config().vocab_size));
      return 1;
    }
    if (!AllFinite(logits)) {
      std::fprintf(stderr, "FAIL [%s]: prefill logits contain NaN/Inf\n", LayoutName(layout));
      return 1;
    }
    if (model.PositionCount() != static_cast<int64_t>(prompt.size())) {
      std::fprintf(stderr, "FAIL [%s]: PositionCount()=%lld, expected %zu\n", LayoutName(layout),
                   static_cast<long long>(model.PositionCount()), prompt.size());
      return 1;
    }

    // A handful of decode steps continuing the prefill's KV/GDN state.
    int32_t tok = 42;
    for (int step = 0; step < 5; ++step) {
      logits = model.DecodeStep(tok);
      if (!AllFinite(logits)) {
        std::fprintf(stderr, "FAIL [%s]: decode step %d logits contain NaN/Inf\n",
                     LayoutName(layout), step);
        return 1;
      }
      // Greedy-ish next id, clamped into a cheap-to-inspect range; exact value is not checked
      // (accuracy is test_gdn_layer/test_attn_layer/test_final_lm_head's job).
      int64_t best = 0;
      float best_v = logits[0];
      for (size_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > best_v) { best_v = logits[i]; best = static_cast<int64_t>(i); }
      }
      tok = static_cast<int32_t>(best % model.Config().vocab_size);
    }
    if (model.PositionCount() != static_cast<int64_t>(prompt.size()) + 5) {
      std::fprintf(stderr, "FAIL [%s]: PositionCount()=%lld after decode, expected %zu\n",
                   LayoutName(layout), static_cast<long long>(model.PositionCount()),
                   prompt.size() + 5);
      return 1;
    }

    // Model::Reset() must exactly reproduce a fresh Model::Load() (see helper's own comment) --
    // `model` above already carries real prefill+decode state at this point, so this genuinely
    // exercises Reset() clearing something, not a no-op on an already-clean Model.
    const double reset_rel = ResetMatchesFreshLoadRelErr(model, opts, prompt, tok);
    const double reset_tol = 1e-4;  // near-exact: identical kernels over identical state, not a
                                     // cross-run/cross-path comparison like equiv_tol below
    if (!(reset_rel < reset_tol)) {
      std::fprintf(stderr, "FAIL [%s]: Model::Reset() vs fresh Load() rel L2=%.4e (tol=%.0e)\n",
                   LayoutName(layout), reset_rel, reset_tol);
      return 1;
    }
    std::fprintf(stderr, "[PASS] layout=%s Model::Reset() matches fresh Load() rel L2=%.4e\n",
                 LayoutName(layout), reset_rel);

    // Value-gated prefill/decode state-handoff equivalence (file header comment): Prefill(prompt)
    // vs Prefill(prompt[:-1]) + DecodeStep(prompt[-1]) must land on the same next-token logits.
    // Measured on the real 4-layer test container, HIP device 1 (2026-09-19): bf16=1.96e-3,
    // mxfp4=6.99e-3, w4a16=2.09e-3, w4a8=3.86e-2 -- tolerances below are those numbers with
    // headroom (same "measured, not aspirational" convention as test_gdn_layer.cpp's kLooseTol),
    // loose enough to pass what was actually observed while still catching a real regression (a
    // broken state handoff -- e.g. this stage's GDN conv-state depth blocker -- lands an order of
    // magnitude higher, not 2-3x higher).
    const double equiv_tol = (layout == Layout::kBf16) ? 1e-2 : 8e-2;
    const double equiv_rel = PrefillDecodeEquivRelErr(opts, prompt);
    if (!(equiv_rel < equiv_tol)) {
      std::fprintf(stderr,
                   "FAIL [%s]: prefill-vs-decode state-handoff equivalence rel L2=%.4e (tol=%.0e)\n",
                   LayoutName(layout), equiv_rel, equiv_tol);
      return 1;
    }
    std::fprintf(stderr, "[PASS] layout=%s prefill-vs-decode equivalence rel L2=%.4e (tol=%.0e)\n",
                 LayoutName(layout), equiv_rel, equiv_tol);

    std::fprintf(stderr, "[PASS] layout=%s prefill(80) + decode(5): finite logits, "
                          "PositionCount=%lld\n",
                 LayoutName(layout), static_cast<long long>(model.PositionCount()));
    ++ran;
  }

  if (ran == 0) {
    std::fprintf(stderr, "FAIL: no layout ran to completion\n");
    return 1;
  }
  return 0;
}
