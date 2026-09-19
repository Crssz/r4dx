// tests/model/test_mtp.cpp -- correctness tests for r4dx::model::Model's MTP self-speculative
// decode (docs/mtp.md), against the 4-layer MTP-enabled test container
// (D:/models/r4dx/qwen38-27b-l4-mtp.r4dx, converted with `r4dx-convert --layers 4 --mtp on
// --layouts bf16,w4a16`).
//
// Two checks, per the task brief:
//  1. CheckVerifyMatchesSequential: VerifyWindow's per-position logits, for a window of candidate
//     tokens constructed to be EXACTLY what a reference sequential decode itself produces (so
//     every one of them is "accepted" by definition), equal that reference's own per-step logits
//     (rel L2 <= 1e-2) at the corresponding position. This is the speculative-verify decode kernel
//     path (attention's decode kernel at q_len>1, GDN's conv_update/recurrent_update over a
//     genuine multi-candidate window with GdnStateManager's per-candidate slot banking) agreeing
//     with the plain sequential T=1 path it must be mathematically equivalent to.
//  2. CheckRejectionRewind: running real MTP self-speculative decode (DecodeStepMtpGreedy) for
//     several rounds on the 4-layer container -- whose drastically truncated depth makes MTP's own
//     drafts agree with the real (equally truncated) model only some of the time, so this
//     organically exercises BOTH full acceptance and partial/total rejection within one run --
//     must produce EXACTLY the same token sequence a fully sequential DecodeStepGreedy run does.
//     A broken rewind (GDN/KV state left reflecting the rejected candidates rather than only the
//     accepted prefix) would make the token generated immediately after a rejection -- and every
//     token after that -- diverge from the sequential reference; this check would then fail.
//
// SKIPs (CTest SKIPPED, not FAILED) if the container is missing, same convention as
// test_forward_smoke.cpp / test_gdn_layer.cpp.
#include <cstdio>
#include <string>
#include <vector>

#include "model.h"
#include "test_common.h"

using namespace r4dx_test;
using r4dx::model::Layout;
using r4dx::model::LayoutName;
using r4dx::model::Model;
using r4dx::model::ModelOptions;

namespace {
const char* kContainerPath = "D:/models/r4dx/qwen38-27b-l4-mtp.r4dx";
constexpr int64_t kDraftK = 3;
constexpr int kRejectionCheckRounds = 12;  // enough rounds to almost certainly see a rejection on
                                            // this drastically-truncated 4-layer container

int32_t Argmax(const std::vector<float>& logits) {
  int64_t best = 0;
  float best_v = logits[0];
  for (size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > best_v) {
      best_v = logits[i];
      best = static_cast<int64_t>(i);
    }
  }
  return static_cast<int32_t>(best);
}

std::vector<int32_t> MakePromptTokens(int n) {
  std::vector<int32_t> ids(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) ids[static_cast<size_t>(i)] = 100 + (i * 41) % 5000;
  return ids;
}

bool CheckVerifyMatchesSequential(const ModelOptions& base_opts,
                                   const std::vector<int32_t>& prompt) {
  ModelOptions ref_opts = base_opts;
  ref_opts.mtp_draft_k = 0;
  Model ref = Model::Load(ref_opts);
  std::vector<float> logits0 = ref.Prefill(prompt);
  const int32_t seed = Argmax(logits0);

  // seq_logits[i] must be "logits produced by sequentially processing candidates[i]" (predicting
  // candidates[i+1]), NOT `logits0` (which precedes candidates[0]=seed even being fed in --
  // logits0 is the PROMPT's own next-token prediction, the computation that PRODUCED seed, not the
  // result of processing it). Off-by-one here silently compares each verify row against the WRONG
  // reference step; kDraftK+1 DecodeStep calls (one per candidate, not kDraftK) keeps the two
  // arrays aligned index-for-index.
  std::vector<int32_t> candidates = {seed};
  std::vector<std::vector<float>> seq_logits;
  int32_t tok = seed;
  for (int64_t i = 0; i < kDraftK + 1; ++i) {
    std::vector<float> l = ref.DecodeStep(tok);
    seq_logits.push_back(l);
    tok = Argmax(l);
    if (i < kDraftK) candidates.push_back(tok);
  }
  // candidates is now [seed, t1, ..., t_kDraftK] (kDraftK+1 entries) -- every candidate after
  // `seed` is precisely what sequential decode itself produced, so every row of this window is
  // "accepted" by construction and directly comparable to seq_logits row-for-row.

  ModelOptions mtp_opts = base_opts;
  mtp_opts.mtp_draft_k = kDraftK;
  Model verify = Model::Load(mtp_opts);
  verify.Prefill(prompt);
  std::vector<float> verify_logits;
  const std::vector<int32_t> preds = verify.VerifyWindow(candidates, &verify_logits);

  const int64_t vocab = verify.Config().vocab_size;
  bool ok = true;
  for (int64_t i = 0; i <= kDraftK; ++i) {
    const std::vector<float> row(verify_logits.begin() + i * vocab,
                                  verify_logits.begin() + (i + 1) * vocab);
    const double rel = RelL2(row, seq_logits[static_cast<size_t>(i)]);
    std::fprintf(stderr, "[mtp] verify-vs-sequential row %lld rel L2=%.4e\n",
                 static_cast<long long>(i), rel);
    if (!(rel <= 1e-2)) {
      std::fprintf(stderr, "FAIL: verify row %lld rel L2=%.4e exceeds 1e-2\n",
                   static_cast<long long>(i), rel);
      ok = false;
    }
    if (i < kDraftK && preds[static_cast<size_t>(i)] != candidates[static_cast<size_t>(i + 1)]) {
      std::fprintf(stderr, "FAIL: verify argmax row %lld = %d, expected %d\n",
                   static_cast<long long>(i), preds[static_cast<size_t>(i)],
                   candidates[static_cast<size_t>(i + 1)]);
      ok = false;
    }
  }
  return ok;
}

bool CheckRejectionRewind(const ModelOptions& base_opts, const std::vector<int32_t>& prompt) {
  ModelOptions ref_opts = base_opts;
  ref_opts.mtp_draft_k = 0;
  Model ref = Model::Load(ref_opts);
  std::vector<float> logits0 = ref.Prefill(prompt);
  int32_t ref_tok = Argmax(logits0);

  ModelOptions mtp_opts = base_opts;
  mtp_opts.mtp_draft_k = kDraftK;
  Model mtp = Model::Load(mtp_opts);
  std::vector<float> mtp_logits0 = mtp.Prefill(prompt);
  int32_t mtp_tok = Argmax(mtp_logits0);

  if (mtp_tok != ref_tok) {
    std::fprintf(stderr, "FAIL: prefill argmax mismatch between mtp/ref Models (%d vs %d) -- "
                          "containers/layout diverged before MTP was even exercised\n",
                 mtp_tok, ref_tok);
    return false;
  }

  std::vector<int32_t> ref_seq, mtp_seq;
  bool saw_rejection = false;
  int32_t next_seed = mtp_tok;
  for (int round = 0; round < kRejectionCheckRounds; ++round) {
    const std::vector<int32_t> emitted = mtp.DecodeStepMtpGreedy(next_seed, kDraftK);
    if (static_cast<int64_t>(emitted.size()) <= kDraftK) saw_rejection = true;
    for (int32_t t : emitted) {
      mtp_seq.push_back(t);
      ref_seq.push_back(Argmax(ref.DecodeStep(ref_tok)));
      ref_tok = ref_seq.back();
    }
    next_seed = emitted.back();
  }

  if (!saw_rejection) {
    std::fprintf(stderr,
                 "FAIL: no round rejected any draft in %d rounds -- this check needs at least one "
                 "rejection to actually exercise state rewind (unexpected on the drastically "
                 "truncated 4-layer container; increase kRejectionCheckRounds if this becomes "
                 "flaky)\n",
                 kRejectionCheckRounds);
    return false;
  }

  if (mtp_seq != ref_seq) {
    std::fprintf(stderr, "FAIL: mtp-decoded sequence diverges from sequential reference\n");
    for (size_t i = 0; i < mtp_seq.size() && i < ref_seq.size(); ++i) {
      if (mtp_seq[i] != ref_seq[i]) {
        std::fprintf(stderr, "  first divergence at index %zu: mtp=%d ref=%d\n", i, mtp_seq[i],
                     ref_seq[i]);
        break;
      }
    }
    return false;
  }

  std::fprintf(stderr,
               "[mtp] rejection-rewind check: %zu tokens generated over %d rounds, saw at least "
               "one rejection, mtp sequence == sequential reference sequence\n",
               mtp_seq.size(), kRejectionCheckRounds);
  return true;
}

// Regression test (task's device-resident-draft-loop pass, step 4 -- "decodes T=1 through a Model
// with mtp_draft_k=3 and compares against mtp_draft_k=0"; the M2 fix pass already covers the
// VerifyWindow-vs-sequential comparison above (CheckVerifyMatchesSequential), so this check is
// deliberately a DIFFERENT contract, not a duplicate: PLAIN Model::DecodeStep (never
// DecodeStepMtpGreedy/VerifyWindow) must produce the same T=1 logits regardless of whether the
// Model was Load()'d with mtp_draft_k=0 or mtp_draft_k>0 -- ModelOptions::mtp_draft_k's own doc
// comment ("0 disables MTP entirely... every decode call degenerate EXACTLY to this Model's
// pre-MTP behavior") is exactly this claim, and this pass's own changes (RunChunk/VerifyWindow now
// branching on Container::EmbedTokensDeviceResident() for the embedding gather, GDN's window bank
// widened to 1+mtp_draft_k slots) are precisely the kind of change that could silently break it for
// an MTP-sized Model driven purely through the plain (non-MTP) decode path -- e.g. a caller like
// r4dx-server that loads with --mtp for later use but issues ordinary requests through
// DecodeStep/DecodeStepGreedy.
bool CheckPlainDecodeUnaffectedByMtpConfig(const ModelOptions& base_opts,
                                            const std::vector<int32_t>& prompt) {
  ModelOptions ref_opts = base_opts;
  ref_opts.mtp_draft_k = 0;
  Model ref = Model::Load(ref_opts);
  std::vector<float> ref_logits = ref.Prefill(prompt);

  ModelOptions mtp_opts = base_opts;
  mtp_opts.mtp_draft_k = kDraftK;
  Model mtp = Model::Load(mtp_opts);
  std::vector<float> mtp_logits = mtp.Prefill(prompt);

  bool ok = true;
  constexpr int kSteps = 8;  // several T=1 DecodeStep calls, never DecodeStepMtpGreedy
  int32_t ref_tok = Argmax(ref_logits);
  int32_t mtp_tok = Argmax(mtp_logits);
  for (int step = -1; step < kSteps; ++step) {
    const double rel = RelL2(ref_logits, mtp_logits);
    std::fprintf(stderr, "[mtp] plain-decode-parity step %d rel L2=%.4e\n", step, rel);
    if (!(rel <= 1e-2)) {
      std::fprintf(stderr,
                   "FAIL: plain DecodeStep diverges between mtp_draft_k=0 and mtp_draft_k=%lld "
                   "Models at step %d (rel L2=%.4e)\n",
                   static_cast<long long>(kDraftK), step, rel);
      ok = false;
    }
    if (ref_tok != mtp_tok) {
      std::fprintf(stderr, "FAIL: plain-decode argmax diverges at step %d: ref=%d mtp=%d\n", step,
                   ref_tok, mtp_tok);
      ok = false;
    }
    if (step + 1 == kSteps) break;
    ref_logits = ref.DecodeStep(ref_tok);
    mtp_logits = mtp.DecodeStep(mtp_tok);  // plain DecodeStep -- MTP's own draft/verify path never
                                            // runs on the mtp_opts Model in this check
    ref_tok = Argmax(ref_logits);
    mtp_tok = Argmax(mtp_logits);
  }
  return ok;
}

// Regression test for the PrimeKv host-staging race (review finding, 2026-09-20): Model::RunChunk
// calls MtpHead::PrimeKv TWICE per chunk with no intervening sync whenever mtp_seed_valid_ is
// already true AND T>1 (the boundary n=1 call, then the within-chunk n=T-1 call) -- this only
// happens starting with the SECOND prefill chunk of a prompt longer than max_chunk_ (64 tokens),
// never the first (mtp_seed_valid_ is false for chunk 0's own boundary call). A prompt <=64 tokens
// (every other check in this file uses MakePromptTokens(24)) never exercises this path at all.
// Before the fix, the two calls' pinned host source buffers aliased, so the FIRST call's async H2D
// upload could read data the SECOND call had already overwritten on the CPU by the time the GPU
// got to it -- corrupting MTP's own KV cache at the chunk boundary position (wrong embedding row
// and/or wrong RoPE position primed there), which silently degrades later acceptance/rejection
// decisions without necessarily crashing. This check drives a multi-chunk prefill (100 tokens: a
// full 64-token chunk plus a 36-token remainder) through real MTP self-speculative decode and
// requires the result to exactly match a fully sequential reference -- same "must be
// byte-identical to sequential DecodeStep" contract as CheckRejectionRewind above, just with a
// prompt long enough to force the two-PrimeKv-calls-per-chunk code path at least once.
bool CheckMultiChunkPrefillPrimeKvRace(const ModelOptions& base_opts) {
  const std::vector<int32_t> prompt = MakePromptTokens(100);  // 100 > 64 == max_chunk_: 2 chunks

  ModelOptions ref_opts = base_opts;
  ref_opts.mtp_draft_k = 0;
  Model ref = Model::Load(ref_opts);
  std::vector<float> logits0 = ref.Prefill(prompt);
  int32_t ref_tok = Argmax(logits0);

  ModelOptions mtp_opts = base_opts;
  mtp_opts.mtp_draft_k = kDraftK;
  Model mtp = Model::Load(mtp_opts);
  std::vector<float> mtp_logits0 = mtp.Prefill(prompt);  // exercises the 2-chunk PrimeKv pairing
  int32_t mtp_tok = Argmax(mtp_logits0);

  if (mtp_tok != ref_tok) {
    std::fprintf(stderr,
                 "FAIL: multi-chunk prefill argmax mismatch between mtp/ref Models (%d vs %d)\n",
                 mtp_tok, ref_tok);
    return false;
  }

  std::vector<int32_t> ref_seq, mtp_seq;
  int32_t next_seed = mtp_tok;
  for (int round = 0; round < kRejectionCheckRounds; ++round) {
    const std::vector<int32_t> emitted = mtp.DecodeStepMtpGreedy(next_seed, kDraftK);
    for (int32_t t : emitted) {
      mtp_seq.push_back(t);
      ref_seq.push_back(Argmax(ref.DecodeStep(ref_tok)));
      ref_tok = ref_seq.back();
    }
    next_seed = emitted.back();
  }

  if (mtp_seq != ref_seq) {
    std::fprintf(stderr,
                 "FAIL: multi-chunk-prefill mtp-decoded sequence diverges from sequential "
                 "reference (PrimeKv host-staging race regression)\n");
    for (size_t i = 0; i < mtp_seq.size() && i < ref_seq.size(); ++i) {
      if (mtp_seq[i] != ref_seq[i]) {
        std::fprintf(stderr, "  first divergence at index %zu: mtp=%d ref=%d\n", i, mtp_seq[i],
                     ref_seq[i]);
        break;
      }
    }
    return false;
  }

  std::fprintf(stderr,
               "[mtp] multi-chunk-prefill PrimeKv-race check: %zu tokens over %d rounds after a "
               "100-token (2-chunk) prefill, mtp sequence == sequential reference sequence\n",
               mtp_seq.size(), kRejectionCheckRounds);
  return true;
}

// Regression test for VerifyWindow's missing precondition check (review finding, 2026-09-20):
// mtp_logits_dev_/mtp_argmax_dev_ are sized for exactly mtp_draft_k_+1 candidates, but the method
// only used to check candidates.size() against max_chunk_ (64) -- a call with more than
// mtp_draft_k_+1 candidates would silently overflow those buffers. VerifyWindow is public
// specifically for tests like this one (its own doc comment says so), so a caller mistake here
// must throw cleanly, not corrupt device memory.
bool CheckVerifyWindowRejectsTooManyCandidates(const ModelOptions& base_opts) {
  ModelOptions mtp_opts = base_opts;
  mtp_opts.mtp_draft_k = kDraftK;  // 3 -- so candidates.size() must be in [1,4]
  Model mtp = Model::Load(mtp_opts);
  mtp.Prefill(MakePromptTokens(24));

  // kDraftK+2 == 5 candidates: within max_chunk_ (64) but over mtp_draft_k_+1 (4).
  std::vector<int32_t> too_many(static_cast<size_t>(kDraftK + 2), 100);
  bool threw = false;
  try {
    mtp.VerifyWindow(too_many);
  } catch (const std::exception&) {
    threw = true;
  }
  if (!threw) {
    std::fprintf(stderr,
                 "FAIL: VerifyWindow accepted %zu candidates on a mtp_draft_k=%lld Model (expected "
                 "a throw -- mtp_logits_dev_/mtp_argmax_dev_ are sized for mtp_draft_k_+1)\n",
                 too_many.size(), static_cast<long long>(kDraftK));
    return false;
  }
  std::fprintf(stderr, "[mtp] VerifyWindow correctly rejected %zu candidates (> mtp_draft_k_+1)\n",
               too_many.size());
  return true;
}

}  // namespace

int main() {
  if (!FileExists(kContainerPath)) {
    return SkipMissing(kContainerPath);
  }

  const Layout layouts[] = {Layout::kBf16, Layout::kW4a16};  // container was converted with
                                                              // --layouts bf16,w4a16 (docs/mtp.md)
  const std::vector<int32_t> prompt = MakePromptTokens(24);
  int ran = 0;
  for (Layout layout : layouts) {
    ModelOptions opts;
    opts.container_path = kContainerPath;
    opts.layout = layout;
    opts.max_ctx = 256;
    opts.layer_limit = 4;

    if (!CheckVerifyMatchesSequential(opts, prompt)) {
      std::fprintf(stderr, "FAIL [%s]: CheckVerifyMatchesSequential\n", LayoutName(layout));
      return 1;
    }
    std::fprintf(stderr, "[PASS] layout=%s CheckVerifyMatchesSequential\n", LayoutName(layout));

    if (!CheckRejectionRewind(opts, prompt)) {
      std::fprintf(stderr, "FAIL [%s]: CheckRejectionRewind\n", LayoutName(layout));
      return 1;
    }
    std::fprintf(stderr, "[PASS] layout=%s CheckRejectionRewind\n", LayoutName(layout));

    if (!CheckPlainDecodeUnaffectedByMtpConfig(opts, prompt)) {
      std::fprintf(stderr, "FAIL [%s]: CheckPlainDecodeUnaffectedByMtpConfig\n", LayoutName(layout));
      return 1;
    }
    std::fprintf(stderr, "[PASS] layout=%s CheckPlainDecodeUnaffectedByMtpConfig\n",
                 LayoutName(layout));

    if (!CheckMultiChunkPrefillPrimeKvRace(opts)) {
      std::fprintf(stderr, "FAIL [%s]: CheckMultiChunkPrefillPrimeKvRace\n", LayoutName(layout));
      return 1;
    }
    std::fprintf(stderr, "[PASS] layout=%s CheckMultiChunkPrefillPrimeKvRace\n", LayoutName(layout));

    if (!CheckVerifyWindowRejectsTooManyCandidates(opts)) {
      std::fprintf(stderr, "FAIL [%s]: CheckVerifyWindowRejectsTooManyCandidates\n", LayoutName(layout));
      return 1;
    }
    std::fprintf(stderr, "[PASS] layout=%s CheckVerifyWindowRejectsTooManyCandidates\n", LayoutName(layout));
    ++ran;
  }

  if (ran == 0) {
    std::fprintf(stderr, "FAIL: no layout ran to completion\n");
    return 1;
  }
  return 0;
}
