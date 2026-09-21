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
#include <random>
#include <string>
#include <vector>

#include "model.h"
#include "mtp_round.hpp"
#include "prefix_state.h"
#include "sampled_equality.hpp"
#include "test_common.h"

using namespace r4dx_test;
using r4dx::model::Layout;
using r4dx::model::LayoutName;
using r4dx::model::Model;
using r4dx::model::ModelOptions;

namespace {
// Review finding, 2026-09-20: this used to point at qwen38-27b-l4-mtp.r4dx (bf16+w4a16 only), so
// CheckVerifyMatchesSequential/CheckRejectionRewind/CheckWideWindowRejectionRewind never ran for
// w4a8/mxfp4 -- precisely the two layouts this milestone's fused-epilogue and GEMM-tuning-table
// passes perturbed. qwen38-27b-l4-allmtp.r4dx (converted `--layers 4 --mtp on --layouts
// bf16,w4a16,w4a8,mxfp4`, docs/status.md's "h_seed drift" section) already exists on disk and was
// previously used only by the non-ctest tool_hseed_drift -- pointed at here instead so every check
// in this file's main loop runs against all four layouts.
const char* kContainerPath = "D:/models/r4dx/qwen38-27b-l4-allmtp.r4dx";
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

// Real end-to-end regression test for the "--chat multi-turn + MTP with a forced mid-round stop"
// gap (docs/status.md Known-gaps: "the underlying bookkeeping fix IS unit-tested via
// PrefixState/mtp_round directly, just not a live end-to-end forced-mid-round-stop scenario").
// Unlike tests/server/test_prefix_state.cpp's TestCommitTracksCommittedNotDisplayedTokens (a
// synthetic vector, proving only PrefixState's own arithmetic), this drives a REAL two-turn
// conversation through a real r4dx::model::Model with real speculative rounds (DecodeStepMtpGreedy,
// or oracle-drafted VerifyWindow+CommitVerifiedWindow rounds -- see "Drafter choice" below), feeding
// their actual output through the SAME r4dx::model::ProcessMtpRound / r4dx::server::PrefixState
// helpers src/cli/main.cpp and src/server/engine.cpp use, with a --max-tokens-equivalent budget
// chosen (by a same-Model, same-seed dry run) to stop display inside a round, one token short of
// what that round committed --
// i.e. the round's own atomic commit genuinely leaves at least one "committed but never displayed"
// token in the model's real state, the exact scenario the mid-round bookkeeping fix exists for.
bool CheckChatMultiTurnMidRoundStop(const ModelOptions& base_opts,
                                     const std::vector<int32_t>& prompt1,
                                     const std::vector<int32_t>& prompt2_user) {
  ModelOptions mtp_opts = base_opts;
  mtp_opts.mtp_draft_k = kDraftK;
  auto is_eos = [](int32_t) { return false; };

  // Drafter choice. The point of this check is a round of >=3 tokens that the Model commits
  // atomically. Real MTP rounds are preferred, but the 4-layer test container's MTP head never
  // agrees with its (equally truncated) target -- measured: every one of 64 rounds x 6 prompts x 4
  // layouts returned exactly 1 token -- so on it a real round can never be >=3 and this check used
  // to take a "skipping" early return every single run. When the real-MTP dry run finds no such
  // round, fall back to an ORACLE drafter: drafts are the target's own sequential greedy
  // continuation (from an independent mtp_draft_k=0 Model), pushed through the same public
  // VerifyWindow + CommitVerifiedWindow pair DecodeStepMtpGreedy's own verify/commit is built on
  // (and that the DFlash2 driver uses). Acceptance is still computed from the real preds, and the
  // committed-vs-displayed state in the Model is exactly what a lucky MTP round would have left.
  std::vector<int32_t> oracle_seq;  // oracle_seq[0] == prefill argmax, [i+1] == i-th decoded token
  bool use_oracle = false;
  auto run_round = [&](Model& m, int32_t seed, size_t emitted) -> std::vector<int32_t> {
    if (!use_oracle) return m.DecodeStepMtpGreedy(seed, kDraftK);
    std::vector<int32_t> candidates{seed};
    for (int64_t j = 1; j <= kDraftK && emitted + static_cast<size_t>(j) < oracle_seq.size(); ++j) {
      candidates.push_back(oracle_seq[emitted + static_cast<size_t>(j)]);
    }
    const std::vector<int32_t> preds = m.VerifyWindow(candidates);
    size_t accepted = 0;
    while (accepted + 1 < candidates.size() && preds[accepted] == candidates[accepted + 1]) ++accepted;
    m.CommitVerifiedWindow(static_cast<int64_t>(accepted) + 1);
    std::vector<int32_t> round(candidates.begin() + 1, candidates.begin() + 1 + static_cast<int64_t>(accepted));
    round.push_back(preds[accepted]);
    return round;
  };

  // Pass 1 (dry run): replay the natural (unbudgeted) round trajectory to find a round that itself
  // returns >=3 tokens, and budget display to stop 2 tokens short of ITS OWN end. ProcessMtpRound
  // commits round[0..size-2] (size-1 tokens) unconditionally, so a budget of size-1 within the round
  // displays exactly what was committed (no gap); size-2 leaves one committed-but-undisplayed token.
  // size==2 cannot work: its size-2 == 0 budget means the real run's `while (remaining > 0)` loop
  // stops on the PREVIOUS round's boundary and never runs this round at all.
  auto probe_stop_after = [&]() -> int64_t {
    Model probe = Model::Load(mtp_opts);
    int32_t next = Argmax(probe.Prefill(prompt1));
    int64_t cumulative = 0;
    for (int round_idx = 0; round_idx < 8; ++round_idx) {
      std::vector<int32_t> round = run_round(probe, next, static_cast<size_t>(cumulative));
      if (round.size() >= 3) return cumulative + static_cast<int64_t>(round.size()) - 2;
      cumulative += static_cast<int64_t>(round.size());
      next = round.back();
    }
    return -1;
  };
  int64_t stop_after = probe_stop_after();
  if (stop_after < 0) {
    ModelOptions seq_opts = base_opts;
    seq_opts.mtp_draft_k = 0;
    Model seq = Model::Load(seq_opts);
    int32_t tok = Argmax(seq.Prefill(prompt1));
    for (int i = 0; i < 8 * (kDraftK + 1) + 1; ++i) {
      oracle_seq.push_back(tok);
      tok = seq.DecodeStepGreedy(tok);
    }
    use_oracle = true;
    stop_after = probe_stop_after();
  }
  if (stop_after < 0) {
    std::fprintf(stderr,
                 "FAIL: CheckChatMultiTurnMidRoundStop: no round of >=3 tokens in 8 rounds even with "
                 "oracle drafts (the target's own sequential greedy continuation) -- VerifyWindow is "
                 "rejecting tokens sequential decode itself produces\n");
    return false;
  }
  std::fprintf(stderr, "[mtp] CheckChatMultiTurnMidRoundStop: drafter=%s stop_after=%lld\n",
               use_oracle ? "oracle (real MTP rounds never reached 3 tokens)" : "mtp",
               static_cast<long long>(stop_after));

  // Turn 1, for real: identical Model construction/prompt/K (greedy + deterministic => reproduces
  // pass 1's own trajectory exactly), this time actually enforcing the max_tokens_remaining budget.
  Model turn = Model::Load(mtp_opts);
  int32_t next = Argmax(turn.Prefill(prompt1));

  r4dx::server::PrefixState prefix;
  prefix.Commit(prompt1, {});  // Prefill() alone already committed prompt1

  std::vector<int32_t> committed_turn1, displayed_turn1;
  int64_t remaining = stop_after;
  bool stopped_mid_round = false;
  while (remaining > 0) {
    std::vector<int32_t> round = run_round(turn, next, displayed_turn1.size());
    committed_turn1.push_back(next);
    r4dx::model::MtpRoundResult outcome = r4dx::model::ProcessMtpRound(round, is_eos, remaining);
    committed_turn1.insert(committed_turn1.end(), outcome.committed.begin(), outcome.committed.end());
    displayed_turn1.insert(displayed_turn1.end(), outcome.displayed.begin(), outcome.displayed.end());
    remaining -= static_cast<int64_t>(outcome.displayed.size());
    if (outcome.hit_max_tokens) {
      stopped_mid_round = outcome.committed.size() > outcome.displayed.size();
      break;
    }
    if (outcome.hit_eos) break;  // not expected (is_eos always false) but keep the loop honest
    next = round.back();
  }
  if (!stopped_mid_round) {
    std::fprintf(stderr,
                 "FAIL: CheckChatMultiTurnMidRoundStop did not land mid-round (stop_after=%lld) -- "
                 "test construction bug, not a product bug\n",
                 static_cast<long long>(stop_after));
    return false;
  }
  prefix.Commit(prompt1, committed_turn1);
  std::fprintf(stderr,
               "[mtp] CheckChatMultiTurnMidRoundStop: turn 1 stopped mid-round: displayed=%zu "
               "committed=%zu (%zu committed-but-undisplayed token(s)), fed_.size()=%zu\n",
               displayed_turn1.size(), committed_turn1.size(),
               committed_turn1.size() - displayed_turn1.size(), prefix.fed().size());

  // Turn 2: a real chat client re-renders the conversation from what was actually DISPLAYED
  // (prompt1 + displayed_turn1), never from the model's internal over-generation.
  std::vector<int32_t> full_tokens_turn2 = prompt1;
  full_tokens_turn2.insert(full_tokens_turn2.end(), displayed_turn1.begin(), displayed_turn1.end());
  full_tokens_turn2.insert(full_tokens_turn2.end(), prompt2_user.begin(), prompt2_user.end());

  // The whole point of the fix: this must NOT look like a valid extension of the model's real
  // (committed) state -- it desyncs on purpose one token before displayed_turn1's own committed
  // tail, so PrefixState must force a Reset()+full-reprefill, not a silently-misaligned fast path.
  if (auto tail = prefix.Extend(full_tokens_turn2); tail.has_value()) {
    std::fprintf(stderr,
                 "FAIL: PrefixState::Extend() wrongly treated a post-mid-round-stop turn 2 as a "
                 "simple extension (tail.size()=%zu) -- this is exactly the silent-desync bug the "
                 "committed-vs-displayed fix exists to prevent\n",
                 tail->size());
    return false;
  }
  std::fprintf(stderr,
               "[mtp] CheckChatMultiTurnMidRoundStop: PrefixState correctly refused the fast-path "
               "extension (forces Reset()+full reprefill, as a real caller must)\n");

  // Recovery path a real caller takes on Extend()==nullopt: Reset() + feed the whole turn-2 prompt
  // from scratch, continue with plain (non-MTP) greedy decode -- exactly main()'s "not an
  // extension" branch. Its result must be byte-identical to an INDEPENDENTLY loaded, mtp_draft_k=0
  // reference Model fed the same turn-2 conversation from scratch (same standard CheckRejectionRewind/
  // CheckMultiChunkPrefillPrimeKvRace already hold this codebase to: bit-exact greedy argmax
  // equality, not just a loose rel-L2 bound).
  turn.Reset();
  prefix.Clear();
  int32_t tok = Argmax(turn.Prefill(full_tokens_turn2));
  constexpr int kTurn2Steps = 6;
  std::vector<int32_t> turn2_sequence;
  for (int i = 0; i < kTurn2Steps; ++i) {
    turn2_sequence.push_back(tok);
    tok = turn.DecodeStepGreedy(tok);
  }

  ModelOptions ref_opts = base_opts;
  ref_opts.mtp_draft_k = 0;
  Model ref = Model::Load(ref_opts);
  int32_t ref_tok = Argmax(ref.Prefill(full_tokens_turn2));
  std::vector<int32_t> ref_sequence;
  for (int i = 0; i < kTurn2Steps; ++i) {
    ref_sequence.push_back(ref_tok);
    ref_tok = ref.DecodeStepGreedy(ref_tok);
  }

  if (turn2_sequence != ref_sequence) {
    std::fprintf(stderr,
                 "FAIL: turn-2 continuation (post-Reset(), post-mid-round-stop) diverges from an "
                 "independently-loaded sequential reference fed the same turn-2 conversation\n");
    for (size_t i = 0; i < turn2_sequence.size() && i < ref_sequence.size(); ++i) {
      if (turn2_sequence[i] != ref_sequence[i]) {
        std::fprintf(stderr, "  first divergence at index %zu: turn2=%d ref=%d\n", i,
                     turn2_sequence[i], ref_sequence[i]);
        break;
      }
    }
    return false;
  }
  std::fprintf(stderr,
               "[mtp] CheckChatMultiTurnMidRoundStop: turn-2 continuation matches the sequential "
               "reference exactly (%d tokens)\n",
               kTurn2Steps);
  return true;
}

// docs/r9700.md R9 ("raise the draft width beyond today's K<=4 ... extend tests/model/test_mtp.cpp
// ... for the wider windows"): the SAME lossless-rewind contract as CheckRejectionRewind above
// (mtp-decoded sequence must exactly equal a sequential reference, even across a rejection), but at
// K=16 -- deliberately re-opening the exact risk class the task calls out ("M3's review found a
// blocker-class off-by-one in exactly this bookkeeping; widening re-opens that risk"): GDN's window
// bank (gdn_state.h), the conv-buffer rolling depth (state_len_max = conv_width-2+max_decode_window),
// and VerifyWindow's mtp_logits_dev_/mtp_argmax_dev_ scratch (sized (draft_k+1)*vocab) all scale
// with mtp_draft_k at Model::Load() time (model.cpp) -- this is the real-hardware regression test
// that those size-K formulas hold at K=16, not just the K=3 this file already covered.
bool CheckWideWindowRejectionRewind(const ModelOptions& base_opts, const std::vector<int32_t>& prompt,
                                     int64_t wide_k) {
  ModelOptions ref_opts = base_opts;
  ref_opts.mtp_draft_k = 0;
  Model ref = Model::Load(ref_opts);
  std::vector<float> logits0 = ref.Prefill(prompt);
  int32_t ref_tok = Argmax(logits0);

  ModelOptions mtp_opts = base_opts;
  mtp_opts.mtp_draft_k = wide_k;
  Model mtp = Model::Load(mtp_opts);
  std::vector<float> mtp_logits0 = mtp.Prefill(prompt);
  int32_t mtp_tok = Argmax(mtp_logits0);

  if (mtp_tok != ref_tok) {
    std::fprintf(stderr,
                 "FAIL: wide-window (K=%lld) prefill argmax mismatch between mtp/ref Models (%d "
                 "vs %d)\n",
                 static_cast<long long>(wide_k), mtp_tok, ref_tok);
    return false;
  }

  std::vector<int32_t> ref_seq, mtp_seq;
  bool saw_rejection = false;
  bool saw_full_acceptance = false;
  int32_t next_seed = mtp_tok;
  const int rounds = kRejectionCheckRounds;
  for (int round = 0; round < rounds; ++round) {
    const std::vector<int32_t> emitted = mtp.DecodeStepMtpGreedy(next_seed, wide_k);
    if (static_cast<int64_t>(emitted.size()) <= wide_k) saw_rejection = true;
    if (static_cast<int64_t>(emitted.size()) == wide_k + 1) saw_full_acceptance = true;
    for (int32_t t : emitted) {
      mtp_seq.push_back(t);
      ref_seq.push_back(Argmax(ref.DecodeStep(ref_tok)));
      ref_tok = ref_seq.back();
    }
    next_seed = emitted.back();
  }

  if (!saw_rejection) {
    std::fprintf(stderr,
                 "FAIL: wide-window (K=%lld) check saw no rejection in %d rounds -- cannot confirm "
                 "state rewind at this width\n",
                 static_cast<long long>(wide_k), rounds);
    return false;
  }

  if (mtp_seq != ref_seq) {
    std::fprintf(stderr,
                 "FAIL: wide-window (K=%lld) mtp-decoded sequence diverges from sequential "
                 "reference\n",
                 static_cast<long long>(wide_k));
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
               "[mtp] wide-window K=%lld check: %zu tokens over %d rounds, saw_rejection=%s "
               "saw_full_acceptance=%s, mtp sequence == sequential reference\n",
               static_cast<long long>(wide_k), mtp_seq.size(), rounds,
               saw_rejection ? "yes" : "no", saw_full_acceptance ? "yes" : "no");
  return true;
}

// docs/r9700.md R9 ("reduced-vocab draft head"): end-to-end lossless check against a container that
// actually has mtp.draft_head.* tensors (D:/models/r4dx/qwen38-27b-l4-mtp-draftvocab.r4dx, converted
// with an arbitrary --draft-vocab-ids test subset of 4096 ids -- coverage is irrelevant to THIS
// check, which is purely about correctness plumbing, not acceptance rate). Runs the SAME
// lossless-rewind contract with the reduced-vocab draft head ENABLED (the default) and again with it
// explicitly disabled, and requires BOTH to exactly equal the sequential reference -- proving the
// technique is lossless (a draft the reduced head could not represent is just a rejected draft,
// never a wrong accepted one) regardless of which path drafted it. Also asserts
// Model::MtpUsingReducedVocabDraft() correctly reports which path is active.
bool CheckReducedVocabDraftHeadLossless(const std::string& container_path, Layout layout,
                                         const std::vector<int32_t>& prompt) {
  ModelOptions base_opts;
  base_opts.container_path = container_path;
  base_opts.layout = layout;
  base_opts.max_ctx = 256;
  base_opts.layer_limit = 4;

  ModelOptions ref_opts = base_opts;
  ref_opts.mtp_draft_k = 0;
  Model ref = Model::Load(ref_opts);
  int32_t ref_seed = Argmax(ref.Prefill(prompt));

  for (bool use_reduced : {true, false}) {
    ModelOptions mtp_opts = base_opts;
    mtp_opts.mtp_draft_k = kDraftK;
    mtp_opts.mtp_draft_reduced_vocab = use_reduced;
    Model mtp = Model::Load(mtp_opts);
    int32_t mtp_seed = Argmax(mtp.Prefill(prompt));

    if (!mtp.MtpEnabled()) {
      std::fprintf(stderr, "FAIL: MtpEnabled() false on a container built with --mtp on\n");
      return false;
    }
    if (mtp.MtpUsingReducedVocabDraft() != use_reduced) {
      std::fprintf(stderr,
                   "FAIL: MtpUsingReducedVocabDraft()=%s, expected %s (container has draft_head.* "
                   "tensors, so 'reduced' should be honored exactly)\n",
                   mtp.MtpUsingReducedVocabDraft() ? "true" : "false", use_reduced ? "true" : "false");
      return false;
    }

    if (mtp_seed != ref_seed) {
      std::fprintf(stderr, "FAIL: prefill argmax mismatch (use_reduced=%s): %d vs %d\n",
                   use_reduced ? "true" : "false", mtp_seed, ref_seed);
      return false;
    }

    // Fresh sequential reference per use_reduced branch (ref's own state must not have advanced
    // past the prefill it shares with mtp above).
    ModelOptions seq_opts = base_opts;
    seq_opts.mtp_draft_k = 0;
    Model seq_ref = Model::Load(seq_opts);
    int32_t seq_tok = Argmax(seq_ref.Prefill(prompt));

    std::vector<int32_t> mtp_seq, seq_seq;
    int32_t next_seed = mtp_seed;
    for (int round = 0; round < kRejectionCheckRounds; ++round) {
      const std::vector<int32_t> emitted = mtp.DecodeStepMtpGreedy(next_seed, kDraftK);
      for (int32_t t : emitted) {
        mtp_seq.push_back(t);
        seq_seq.push_back(Argmax(seq_ref.DecodeStep(seq_tok)));
        seq_tok = seq_seq.back();
      }
      next_seed = emitted.back();
    }

    if (mtp_seq != seq_seq) {
      std::fprintf(stderr,
                   "FAIL: reduced-vocab-draft-head check (use_reduced=%s) diverges from sequential "
                   "reference\n",
                   use_reduced ? "true" : "false");
      for (size_t i = 0; i < mtp_seq.size() && i < seq_seq.size(); ++i) {
        if (mtp_seq[i] != seq_seq[i]) {
          std::fprintf(stderr, "  first divergence at index %zu: mtp=%d ref=%d\n", i, mtp_seq[i],
                       seq_seq[i]);
          break;
        }
      }
      return false;
    }
    std::fprintf(stderr,
                 "[mtp] reduced-vocab-draft-head check (use_reduced=%s, layout=%s): %zu tokens "
                 "over %d rounds, mtp sequence == sequential reference\n",
                 use_reduced ? "true" : "false", LayoutName(layout), mtp_seq.size(),
                 kRejectionCheckRounds);
  }
  return true;
}

// ================================================================================================
// Milestone 6 stage S2: sampled speculative rounds (docs/sampling.md section 9)
// ================================================================================================
// The losslessness gate. A sampled MTP round draws exactly ONE uniform per EMITTED token and maps
// it to a token by the same canonical rule plain sampled decode uses, so for a fixed seed the two
// must emit the SAME tokens in the same order -- not merely the same distribution. Everything below
// is that equality, at three sampling configurations x three seeds x all four layouts, against an
// independently loaded mtp_draft_k=0 Model.
//
// Two drafters are exercised, because this 4-layer container's MTP head hardly ever agrees with its
// (equally truncated) target:
//   * the REAL MTP head -- mostly the ALL-REJECTED path (a round is usually one token, i.e. the
//     round's row 0 sample must equal what plain decode would have sampled);
//   * an ORACLE drafter whose drafts are the plain sampled run's OWN tokens -- the ALL-ACCEPTED
//     path (rounds are k+1 tokens long), which is where a bookkeeping error in the commit count or
//     in the per-row draw order would actually show up.
//
// THE ONE DIVERGENCE THIS FILE ACCEPTS, and why it is not a loosening. A speculative round computes
// its logits in ONE q_len>1 forward pass; plain decode computes them one row at a time. Those are
// different GEMM shapes, so their reduction order differs, and floating-point addition is not
// associative -- CheckVerifyMatchesSequential above measures the result directly on this very
// container and accepts it up to rel L2 <= 1e-2 (measured: ~1.1e-3 at bf16). A greedy argmax
// survives that; a CDF walk does not always, because the same 1e-3 perturbation moves a candidate
// boundary by ~1e-2 of the distribution's mass, and a draw landing inside that band picks the
// neighbouring token. This is the SAME "known batched-verify divergence" class
// tools/validate_dflash.ps1 exists to adjudicate, and it is handled the same way: never silently.
// On any mismatch, ClassifySampledDivergence below obtains BOTH logits rows for the diverging
// position -- the single-row decode one (by replaying the plain trajectory) and the EXACT verify
// row the speculative sampler resolved that token from (by re-running the deterministic speculative
// trajectory and reading it back with Model::ReadVerifyLogitsRow) -- and the divergence is accepted
// ONLY if all of these hold, each of which a real bookkeeping bug would break:
//   1. canonically sampling the DECODE row with that token's own draw u reproduces the plain run's
//      token, so the reference trajectory really is canonical;
//   2. canonically sampling the EXACT VERIFY row with that SAME u reproduces the SPECULATIVE run's
//      token -- which pins the draw index, the window row, and the filters all at once: had the
//      round sampled a different row, used a shifted draw, or applied the filters differently, the
//      row this emission index maps to would not reproduce what it emitted;
//   3. the two rows are the same next-token distribution (a sanity floor on rel L2 -- see the note
//      in the classifier for why this is deliberately loose while 1 and 2 are exact).
// Independently of all that, every runner below asserts the exact position-lockstep invariant after
// EVERY round (a round commits exactly as many positions as it emitted tokens), which is the
// bookkeeping gate proper. Everything else fails, loudly, with every number printed.
constexpr size_t kSampledTokens = 48;
constexpr uint64_t kSampledSeeds[] = {1u, 20260921u, 0x9E3779B97F4A7C15ull};

// Plain sampled decode: the reference trajectory. One draw for the prefill row's own token, then
// one per DecodeStepSampled call.
std::vector<int32_t> RunPlainSampled(Model& m, const std::vector<int32_t>& prompt,
                                      const r4dx::kernels::SampleParams& params, uint64_t seed,
                                      size_t n) {
  m.Reset();
  std::mt19937_64 rng = r4dx::kernels::MakeRng(seed);
  const std::vector<float> l0 = m.Prefill(prompt);
  std::vector<int32_t> seq;
  int32_t tok = SampleFirstToken(l0, m.Config().vocab_size, params, rng);
  seq.push_back(tok);
  while (seq.size() < n) {
    tok = m.DecodeStepSampled(tok, params, rng);
    seq.push_back(tok);
  }
  // The invariant every runner in this file asserts (see RunMtpSampled): after N emitted tokens the
  // model has committed the prompt plus N-1 of them (the last is the next call's anchor).
  if (m.PositionCount() != static_cast<int64_t>(prompt.size() + seq.size() - 1)) {
    std::fprintf(stderr, "FAIL: plain sampled decode committed %lld positions, expected %zu\n",
                 static_cast<long long>(m.PositionCount()), prompt.size() + seq.size() - 1);
    std::abort();
  }
  return seq;
}

// The same trajectory through real MTP self-speculative sampled rounds.
//
// `capture_index`/`captured_row`: forensics for a mismatch. On a re-run (the trajectory is
// deterministic in the seed) this reads back, through Model::ReadVerifyLogitsRow, the EXACT fp32
// logits row the sampler resolved emitted token `capture_index` from -- valid because
// verify_logits_dev_ still holds that round's window when DecodeStepMtpSampled returns and nothing
// touches it until the next round. Having the real row, rather than a reconstruction of it, is what
// makes ClassifySampledDivergence's verdict threshold-free.
std::vector<int32_t> RunMtpSampled(Model& m, const std::vector<int32_t>& prompt,
                                    const r4dx::kernels::SampleParams& params, uint64_t seed,
                                    size_t n, int64_t k, size_t* longest_round_out,
                                    size_t capture_index = static_cast<size_t>(-1),
                                    std::vector<float>* captured_row = nullptr) {
  m.Reset();
  std::mt19937_64 rng = r4dx::kernels::MakeRng(seed);
  const std::vector<float> l0 = m.Prefill(prompt);
  std::vector<int32_t> seq;
  int32_t tok = SampleFirstToken(l0, m.Config().vocab_size, params, rng);
  seq.push_back(tok);
  size_t longest = 0;
  while (seq.size() < n) {
    const size_t base = seq.size();  // global index of this round's first emitted token
    const std::vector<int32_t> round = m.DecodeStepMtpSampled(tok, k, params, rng);
    if (captured_row != nullptr && capture_index >= base && capture_index < base + round.size()) {
      m.ReadVerifyLogitsRow(static_cast<int64_t>(capture_index - base), *captured_row);
    }
    longest = std::max(longest, round.size());
    seq.insert(seq.end(), round.begin(), round.end());
    tok = round.back();
    // THE bookkeeping gate, asserted continuously rather than only on a mismatch: a round commits
    // exactly as many positions as it emitted tokens, so after N emitted tokens the model holds the
    // prompt plus N-1 of them -- identical to plain decode's own arithmetic. A wrong commit count
    // (the classic speculative-decoding bug: committing the whole window, or one row too few) is
    // caught here immediately, on every round, with no tolerance and no forensics needed.
    if (m.PositionCount() != static_cast<int64_t>(prompt.size() + seq.size() - 1)) {
      std::fprintf(stderr,
                   "FAIL: after a sampled MTP round of %zu tokens the model holds %lld positions, "
                   "expected %zu (prompt %zu + %zu emitted - 1)\n",
                   round.size(), static_cast<long long>(m.PositionCount()),
                   prompt.size() + seq.size() - 1, prompt.size(), seq.size());
      std::abort();
    }
  }
  if (longest_round_out) *longest_round_out = longest;
  return seq;
}

// The same trajectory through ORACLE-drafted sampled rounds: the drafts are `oracle`'s own next k
// tokens, so every draft is accepted and every round is k+1 long -- as long as sample-and-match is
// right. `oracle` must hold at least n + k tokens.
std::vector<int32_t> RunOracleSampled(Model& m, const std::vector<int32_t>& prompt,
                                       const r4dx::kernels::SampleParams& params, uint64_t seed,
                                       size_t n, int64_t k, const std::vector<int32_t>& oracle,
                                       size_t* longest_round_out, OracleRoundStats* stats,
                                       size_t capture_index = static_cast<size_t>(-1),
                                       std::vector<float>* captured_row = nullptr) {
  m.Reset();
  std::mt19937_64 rng = r4dx::kernels::MakeRng(seed);
  const std::vector<float> l0 = m.Prefill(prompt);
  const int64_t vocab = m.Config().vocab_size;
  std::vector<int32_t> seq;
  int32_t tok = SampleFirstToken(l0, vocab, params, rng);
  seq.push_back(tok);
  size_t longest = 0;
  while (seq.size() < n) {
    std::vector<int32_t> drafts;
    for (int64_t j = 0; j < k && seq.size() + static_cast<size_t>(j) < oracle.size(); ++j) {
      drafts.push_back(oracle[seq.size() + static_cast<size_t>(j)]);
    }
    const size_t base = seq.size();
    std::vector<float> window_logits;
    const std::vector<int32_t> round = RunSampledOracleRound(
        m, tok, drafts, params, rng, stats, captured_row ? &window_logits : nullptr);
    if (captured_row != nullptr && capture_index >= base && capture_index < base + round.size()) {
      const size_t row = capture_index - base;
      captured_row->assign(window_logits.begin() + static_cast<ptrdiff_t>(row) * vocab,
                           window_logits.begin() + static_cast<ptrdiff_t>(row + 1) * vocab);
    }
    longest = std::max(longest, round.size());
    seq.insert(seq.end(), round.begin(), round.end());
    tok = round.back();
    if (m.PositionCount() != static_cast<int64_t>(prompt.size() + seq.size() - 1)) {  // see above
      std::fprintf(stderr,
                   "FAIL: after an oracle-drafted sampled round of %zu tokens the model holds %lld "
                   "positions, expected %zu\n",
                   round.size(), static_cast<long long>(m.PositionCount()),
                   prompt.size() + seq.size() - 1);
      std::abort();
    }
  }
  if (longest_round_out) *longest_round_out = longest;
  return seq;
}

// Plain sampled decode over the FULL fp32 logits row -- the definition the summary path must match
// exactly (same Model, same rows, so there is no numeric excuse available here at all).
std::vector<int32_t> RunPlainSampledFullVocab(Model& m, const std::vector<int32_t>& prompt,
                                               const r4dx::kernels::SampleParams& params,
                                               uint64_t seed, size_t n) {
  m.Reset();
  std::mt19937_64 rng = r4dx::kernels::MakeRng(seed);
  std::vector<float> row = m.Prefill(prompt);
  const int64_t vocab = m.Config().vocab_size;
  std::vector<int32_t> seq;
  int32_t tok = r4dx::kernels::SampleCanonical(row.data(), vocab, params,
                                                r4dx::kernels::DrawUniform01(rng));
  seq.push_back(tok);
  while (seq.size() < n) {
    row = m.DecodeStep(tok);
    tok = r4dx::kernels::SampleCanonical(row.data(), vocab, params,
                                          r4dx::kernels::DrawUniform01(rng));
    seq.push_back(tok);
  }
  return seq;
}

// Recomputes the FULL fp32 logits row PLAIN single-row decode used for emitted token `index`, by
// replaying the same trajectory on a fresh (mtp_draft_k=0) Model. Forensics only.
std::vector<float> PlainLogitsRowAt(Model& ref, const std::vector<int32_t>& prompt,
                                     const std::vector<int32_t>& seq, size_t index) {
  ref.Reset();
  std::vector<float> row = ref.Prefill(prompt);
  for (size_t i = 0; i + 1 <= index; ++i) row = ref.DecodeStep(seq[i]);
  return row;  // index==0 -> the prefill row itself
}

enum class DivergenceVerdict { kEqual, kProvenVerifyNumeric, kBug };

// See this section's own header comment for the three conditions. `fetch_verify_row(j)` must return
// the EXACT fp32 logits row the speculative run resolved its emitted token `j` from (the runners
// above re-run the deterministic trajectory and read it back with Model::ReadVerifyLogitsRow) --
// having the real row, not a reconstruction, is what lets condition 3 be an equality rather than a
// tolerance. Returns kBug unless every condition holds.
DivergenceVerdict ClassifySampledDivergence(
    const char* tag, Model& ref, const std::vector<int32_t>& prompt,
    const std::vector<int32_t>& plain, const std::vector<int32_t>& spec, uint64_t seed,
    const r4dx::kernels::SampleParams& params,
    const std::function<std::vector<float>(size_t)>& fetch_verify_row) {
  size_t j = 0;
  while (j < plain.size() && j < spec.size() && plain[j] == spec[j]) ++j;
  if (j >= kSampledTokens) return DivergenceVerdict::kEqual;

  const int64_t vocab = ref.Config().vocab_size;
  const std::vector<double> draws = ReplayDraws(seed, j + 1);
  const double u = draws[j];
  std::fprintf(stderr,
               "%s DIVERGENCE at index %zu after %zu identical tokens: plain=%d spec=%d u=%.17g\n",
               tag, j, j, plain[j], spec[j], u);
  if (j == 0) {
    std::fprintf(stderr,
                 "%s   index 0 comes from Prefill()'s own logits on BOTH sides, with no verify pass "
                 "anywhere -- a divergence here cannot be the batched-verify mechanism\n",
                 tag);
    return DivergenceVerdict::kBug;
  }

  const std::vector<float> row_decode = PlainLogitsRowAt(ref, prompt, plain, j);
  const std::vector<float> row_verify = fetch_verify_row(j);
  if (row_verify.size() != static_cast<size_t>(vocab)) {
    std::fprintf(stderr, "%s   could not capture the verify row for index %zu\n", tag, j);
    return DivergenceVerdict::kBug;
  }
  const double rel = RelL2(row_verify, row_decode);
  const int32_t from_decode = r4dx::kernels::SampleCanonical(row_decode.data(), vocab, params, u);
  const int32_t from_verify = r4dx::kernels::SampleCanonical(row_verify.data(), vocab, params, u);
  const NearTieReport tie_decode = AnalyzeNearTie(row_decode, vocab, params, u, plain[j], spec[j]);
  const NearTieReport tie_verify = AnalyzeNearTie(row_verify, vocab, params, u, plain[j], spec[j]);
  std::fprintf(stderr,
               "%s   decode-row vs verify-row rel L2=%.4e; canonical sample of the decode row at "
               "that u=%d (plain emitted %d); of the verify row=%d (spec emitted %d)\n",
               tag, rel, from_decode, plain[j], from_verify, spec[j]);
  PrintNearTie(tag, tie_decode, u, plain[j], spec[j]);
  // The whole mechanism in two numbers: the CDF boundary between these two candidates sits at
  // `boundary` in the decode row and at a slightly different place in the verify row, and `u` lies
  // BETWEEN the two -- which is exactly "a near-tie in the post-filter probabilities straddling u".
  std::fprintf(stderr,
               "%s   that pair's CDF boundary: decode row %.12f, verify row %.12f (shift %.3e), "
               "u=%.12f lies %s the two\n",
               tag, tie_decode.boundary, tie_verify.boundary,
               tie_verify.boundary - tie_decode.boundary, u,
               (u >= std::min(tie_decode.boundary, tie_verify.boundary) &&
                u < std::max(tie_decode.boundary, tie_verify.boundary))
                   ? "BETWEEN"
                   : "outside");

  // Condition 3's sanity floor. `rel` is NOT required to be tiny: the two trajectories' GDN/KV
  // state has been drifting apart in its last bits since the very first speculative round, and that
  // drift accumulates, so by token 30 of a w4a8 run the two rows can differ by a few percent while
  // still being the same next-token distribution for the same context. What a WRONG-CONTEXT bug
  // (an off-by-one commit, the wrong window row) would produce is not a few percent -- it is two
  // unrelated distributions, i.e. rel L2 near sqrt(2). This bound only has to separate those two
  // regimes, and the bookkeeping itself is pinned exactly elsewhere (the position-lockstep
  // assertion in every runner above, plus condition 2 below, which fails outright if the row the
  // sampler used was not the row this emission index maps to).
  const bool cond_rows_same_context = (rel > 0.0 && rel <= 0.5);
  const bool cond_decode_reproduces_plain = (from_decode == plain[j]);
  const bool cond_verify_reproduces_spec = (from_verify == spec[j]);
  if (cond_rows_same_context && cond_decode_reproduces_plain && cond_verify_reproduces_spec) {
    std::fprintf(stderr,
                 "%s   ACCEPTED as the known batched-verify divergence (docs/perf.md, "
                 "tools/validate_dflash.ps1): the speculative path sampled the RIGHT draw, from the "
                 "RIGHT row of the RIGHT round, with the RIGHT filters -- the only difference is "
                 "that row's own last bits, by the q_len>1-vs-T=1 reduction-order mechanism "
                 "CheckVerifyMatchesSequential already measures on this container\n",
                 tag);
    return DivergenceVerdict::kProvenVerifyNumeric;
  }
  std::fprintf(stderr,
               "%s   NOT the known mechanism (rows-are-the-same-distribution=%s "
               "decode-row-reproduces-plain=%s verify-row-reproduces-spec=%s) -- this is a "
               "bookkeeping bug\n",
               tag, cond_rows_same_context ? "yes" : "NO",
               cond_decode_reproduces_plain ? "yes" : "NO",
               cond_verify_reproduces_spec ? "yes" : "NO");
  return DivergenceVerdict::kBug;
}

// Stage S2 item 1: the SUMMARY path itself, with no speculation anywhere and therefore no numeric
// excuse available. DecodeStepSampled resolves each token from the device row summary
// (r4dx_topk_lse_f32's top-64 + logsumexp, 516 bytes); DecodeStep + SampleCanonical resolves it
// from the same step's full 993 KB fp32 row. Same Model, same rows, same draws, so the two
// trajectories must be EXACTLY equal -- this is stage S1's exactness claim re-tested on real model
// logits rather than synthetic ones, end to end through the device kernel.
bool CheckSampledSummaryPathExact(const ModelOptions& base_opts, const std::vector<int32_t>& prompt,
                                   Layout layout) {
  ModelOptions opts = base_opts;
  opts.mtp_draft_k = 0;
  Model m = Model::Load(opts);
  // One seed per configuration, not the full seed sweep: every oracle-drafted round in
  // CheckSampledRoundsMatchPlain below ALSO cross-checks the summary path row by row (see
  // RunSampledOracleRound's own summary-vs-full-vocab assertion), across all three seeds, so a
  // second seed here would only re-pay a real trajectory for coverage already obtained.
  for (const SampledConfig& cfg : SampledConfigs()) {
    {
      const uint64_t seed = kSampledSeeds[0];
      const int64_t before = m.SampledFallbackRows();
      const std::vector<int32_t> summary_path =
          RunPlainSampled(m, prompt, cfg.params, seed, kSampledTokens);
      const int64_t fallbacks = m.SampledFallbackRows() - before;
      const std::vector<int32_t> full_path =
          RunPlainSampledFullVocab(m, prompt, cfg.params, seed, kSampledTokens);
      if (summary_path != full_path) {
        size_t j = 0;
        while (j < summary_path.size() && summary_path[j] == full_path[j]) ++j;
        std::fprintf(stderr,
                     "FAIL: DecodeStepSampled (device row summary) disagrees with DecodeStep + the "
                     "full-vocab canonical sampler on the SAME Model at index %zu (summary=%d "
                     "full=%d, layout=%s config=%s seed=%llu) -- the summary path is not exact\n",
                     j, summary_path[j], full_path[j], LayoutName(layout), cfg.name,
                     static_cast<unsigned long long>(seed));
        return false;
      }
      std::fprintf(stderr,
                   "[mtp] summary-path-exact layout=%s config=%-20s seed=%llu: %zu tokens "
                   "identical to the full-vocab path (%lld/%zu rows fell back)\n",
                   LayoutName(layout), cfg.name, static_cast<unsigned long long>(seed),
                   kSampledTokens, static_cast<long long>(fallbacks), kSampledTokens);
    }
  }
  return true;
}

bool CheckSampledRoundsMatchPlain(const ModelOptions& base_opts, const std::vector<int32_t>& prompt,
                                   Layout layout) {
  ModelOptions ref_opts = base_opts;
  ref_opts.mtp_draft_k = 0;
  Model ref = Model::Load(ref_opts);

  ModelOptions mtp_opts = base_opts;
  mtp_opts.mtp_draft_k = kDraftK;
  Model mtp = Model::Load(mtp_opts);

  size_t oracle_rounds_over_one = 0;
  size_t exact = 0, accepted_numeric = 0, combos = 0;
  for (const SampledConfig& cfg : SampledConfigs()) {
    for (uint64_t seed : kSampledSeeds) {
      // n + kDraftK tokens so the oracle drafter below always has a full window of kDraftK drafts
      // available, right up to the last compared token.
      const std::vector<int32_t> plain =
          RunPlainSampled(ref, prompt, cfg.params, seed, kSampledTokens + kDraftK);

      size_t longest_mtp = 0;
      const std::vector<int32_t> spec =
          RunMtpSampled(mtp, prompt, cfg.params, seed, kSampledTokens, kDraftK, &longest_mtp);
      if (spec.size() < kSampledTokens) {
        std::fprintf(stderr, "FAIL: MTP sampled run produced only %zu tokens\n", spec.size());
        return false;
      }

      OracleRoundStats stats;
      size_t longest_oracle = 0;
      const std::vector<int32_t> oracle_run = RunOracleSampled(
          mtp, prompt, cfg.params, seed, kSampledTokens, kDraftK, plain, &longest_oracle, &stats);
      if (stats.summary_mismatch) {
        std::fprintf(stderr,
                     "FAIL: a device row summary resolved a row differently from the full-vocab "
                     "canonical sampler on that row's own logits (layout=%s config=%s)\n",
                     LayoutName(layout), cfg.name);
        return false;
      }
      if (longest_oracle > 1) ++oracle_rounds_over_one;

      const std::string base_tag = std::string(LayoutName(layout)) + " " + cfg.name +
                                    " seed=" + std::to_string(seed) + "]";
      // On a divergence, re-run the (deterministic) speculative trajectory once more, this time
      // reading back the exact verify row the diverging token was resolved from.
      auto mtp_row = [&](size_t j) {
        std::vector<float> row;
        RunMtpSampled(mtp, prompt, cfg.params, seed, kSampledTokens, kDraftK, nullptr, j, &row);
        return row;
      };
      auto oracle_row = [&](size_t j) {
        std::vector<float> row;
        RunOracleSampled(mtp, prompt, cfg.params, seed, kSampledTokens, kDraftK, plain, nullptr,
                         nullptr, j, &row);
        return row;
      };
      const std::pair<std::string, const std::vector<int32_t>*> runs[2] = {
          {"[mtp-sampled " + base_tag, &spec}, {"[mtp-sampled-oracle " + base_tag, &oracle_run}};
      const std::function<std::vector<float>(size_t)> fetchers[2] = {mtp_row, oracle_row};
      for (int which = 0; which < 2; ++which) {
        ++combos;
        const DivergenceVerdict v =
            ClassifySampledDivergence(runs[which].first.c_str(), ref, prompt, plain,
                                      *runs[which].second, seed, cfg.params, fetchers[which]);
        if (v == DivergenceVerdict::kBug) return false;
        if (v == DivergenceVerdict::kEqual) ++exact;
        else ++accepted_numeric;
      }

      std::fprintf(stderr,
                   "[mtp] sampled-equality layout=%s config=%-20s seed=%llu: mtp longest round=%zu, "
                   "oracle longest round=%zu, %lld/%lld oracle summary rows fell back to full vocab\n",
                   LayoutName(layout), cfg.name, static_cast<unsigned long long>(seed), longest_mtp,
                   longest_oracle, static_cast<long long>(stats.fallbacks),
                   static_cast<long long>(stats.rows));
    }
  }

  // The oracle variant exists precisely to exercise rounds LONGER than one token (this container's
  // real MTP head hardly ever accepts). If none ever got past one token, the all-accepted path was
  // not tested at all and the check is worthless -- fail rather than report a green that means
  // nothing.
  if (oracle_rounds_over_one == 0) {
    std::fprintf(stderr,
                 "FAIL: no oracle-drafted sampled round ever emitted more than one token, so the "
                 "all-drafts-accepted path was never exercised (layout=%s)\n",
                 LayoutName(layout));
    return false;
  }
  // A run where EVERY trajectory diverged would mean the accepted-divergence path had become the
  // norm rather than the exception, which is worth failing on even though each individual
  // divergence was proven: it would point at a systematically different row, not at last-bit noise.
  if (exact == 0) {
    std::fprintf(stderr,
                 "FAIL: not one of the %zu sampled trajectories was token-for-token identical to "
                 "plain sampled decode on layout=%s -- every single one was 'explained', which is "
                 "not a green result\n",
                 combos, LayoutName(layout));
    return false;
  }
  std::fprintf(stderr,
               "[mtp] sampled-equality layout=%s: %zu/%zu trajectories token-for-token identical, "
               "%zu accepted as the known batched-verify divergence, 0 bookkeeping failures; %zu of "
               "%zu (config,seed) combinations saw oracle rounds longer than one token\n",
               LayoutName(layout), exact, combos, accepted_numeric, oracle_rounds_over_one,
               SampledConfigs().size() * (sizeof(kSampledSeeds) / sizeof(kSampledSeeds[0])));
  return true;
}

// Stage S2 item 3c: the committed-vs-displayed bookkeeping (mtp_round.hpp's ProcessMtpRound) with a
// SAMPLED round. Same structure as CheckChatMultiTurnMidRoundStop above -- and the same oracle
// drafter, for the same reason (a real MTP round on this container is always exactly one token, and
// a one-token round can never stop "inside" itself) -- but every emitted token now comes from
// sample-and-match rather than from an argmax, and the turn-2 continuation is compared against an
// independently loaded reference driven by DecodeStepSampled.
bool CheckSampledMidRoundStop(const ModelOptions& base_opts, const std::vector<int32_t>& prompt1,
                               const std::vector<int32_t>& prompt2_user) {
  const r4dx::kernels::SampleParams params = SampledConfigs()[0].params;  // T=0.7 top_k=20 top_p=0.8
  constexpr uint64_t kSeed = 4242u;
  constexpr size_t kProbeTokens = 32;
  auto is_eos = [](int32_t) { return false; };

  ModelOptions ref_opts = base_opts;
  ref_opts.mtp_draft_k = 0;
  Model ref = Model::Load(ref_opts);
  ModelOptions mtp_opts = base_opts;
  mtp_opts.mtp_draft_k = kDraftK;
  Model turn = Model::Load(mtp_opts);

  // The oracle drafts: THIS Model's own plain sampled continuation of prompt1 under this seed --
  // taken from `turn` rather than from `ref` on purpose, because this check is about the
  // committed-vs-displayed bookkeeping of a long round, not about cross-Model equality (that is
  // CheckSampledRoundsMatchPlain's job), and drafting from the same Model guarantees the rounds
  // below really do accept and therefore really are >=3 tokens long.
  const std::vector<int32_t> oracle = RunPlainSampled(turn, prompt1, params, kSeed, kProbeTokens);

  // Dry run: find a round of >=3 tokens and budget display to stop 2 tokens short of ITS end, so
  // ProcessMtpRound's unconditional `committed` (round.size()-1 tokens) genuinely exceeds
  // `displayed`. Identical construction to the greedy check above.
  auto replay = [&](int64_t budget, std::vector<int32_t>* committed, std::vector<int32_t>* displayed,
                     bool* stopped_mid_round) -> int64_t {
    turn.Reset();
    std::mt19937_64 rng = r4dx::kernels::MakeRng(kSeed);
    const std::vector<float> l0 = turn.Prefill(prompt1);
    int32_t next = SampleFirstToken(l0, turn.Config().vocab_size, params, rng);
    int64_t emitted = 0;      // tokens the (unbudgeted) dry run has emitted so far
    int64_t remaining = budget;
    int64_t found_stop_after = -1;
    for (int round_idx = 0; round_idx < 8; ++round_idx) {
      if (budget >= 0 && remaining <= 0) break;
      std::vector<int32_t> drafts;
      for (int64_t j = 1; j <= kDraftK && static_cast<size_t>(emitted + j) < oracle.size(); ++j) {
        drafts.push_back(oracle[static_cast<size_t>(emitted + j)]);
      }
      const std::vector<int32_t> round =
          RunSampledOracleRound(turn, next, drafts, params, rng, nullptr);
      if (budget < 0) {  // dry run: just look for a round we could stop inside
        if (round.size() >= 3 && found_stop_after < 0) {
          found_stop_after = emitted + static_cast<int64_t>(round.size()) - 2;
        }
        emitted += static_cast<int64_t>(round.size());
        next = round.back();
        if (found_stop_after >= 0) break;
        continue;
      }
      committed->push_back(next);
      r4dx::model::MtpRoundResult outcome = r4dx::model::ProcessMtpRound(round, is_eos, remaining);
      committed->insert(committed->end(), outcome.committed.begin(), outcome.committed.end());
      displayed->insert(displayed->end(), outcome.displayed.begin(), outcome.displayed.end());
      remaining -= static_cast<int64_t>(outcome.displayed.size());
      emitted += static_cast<int64_t>(outcome.displayed.size());
      if (outcome.hit_max_tokens) {
        *stopped_mid_round = outcome.committed.size() > outcome.displayed.size();
        break;
      }
      if (outcome.hit_eos) break;
      next = round.back();
    }
    return found_stop_after;
  };

  std::vector<int32_t> dummy_c, dummy_d;
  bool dummy_mid = false;
  const int64_t stop_after = replay(-1, &dummy_c, &dummy_d, &dummy_mid);
  if (stop_after < 1) {
    std::fprintf(stderr,
                 "FAIL: CheckSampledMidRoundStop: no oracle-drafted SAMPLED round of >=3 tokens in "
                 "8 rounds -- sample-and-match is rejecting tokens plain sampled decode itself "
                 "produced, which is exactly the losslessness failure this milestone is about\n");
    return false;
  }

  std::vector<int32_t> committed_turn1, displayed_turn1;
  bool stopped_mid_round = false;
  replay(stop_after, &committed_turn1, &displayed_turn1, &stopped_mid_round);
  if (!stopped_mid_round) {
    std::fprintf(stderr,
                 "FAIL: CheckSampledMidRoundStop did not land mid-round (stop_after=%lld) -- test "
                 "construction bug, not a product bug\n",
                 static_cast<long long>(stop_after));
    return false;
  }
  std::fprintf(stderr,
               "[mtp] CheckSampledMidRoundStop: turn 1 stopped mid-round: displayed=%zu "
               "committed=%zu (%zu committed-but-undisplayed token(s))\n",
               displayed_turn1.size(), committed_turn1.size(),
               committed_turn1.size() - displayed_turn1.size());

  r4dx::server::PrefixState prefix;
  prefix.Commit(prompt1, committed_turn1);
  std::vector<int32_t> full_tokens_turn2 = prompt1;
  full_tokens_turn2.insert(full_tokens_turn2.end(), displayed_turn1.begin(), displayed_turn1.end());
  full_tokens_turn2.insert(full_tokens_turn2.end(), prompt2_user.begin(), prompt2_user.end());
  if (auto tail = prefix.Extend(full_tokens_turn2); tail.has_value()) {
    std::fprintf(stderr,
                 "FAIL: CheckSampledMidRoundStop: PrefixState::Extend() wrongly treated a "
                 "post-mid-round-stop turn 2 as a simple extension (tail.size()=%zu)\n",
                 tail->size());
    return false;
  }

  // Recovery: Reset() + full re-prefill + plain SAMPLED continuation, byte-identical to an
  // independently loaded reference fed the same turn-2 conversation with the same seed.
  constexpr size_t kTurn2Tokens = 8;
  constexpr uint64_t kTurn2Seed = 777u;
  const std::vector<int32_t> turn2 =
      RunPlainSampled(turn, full_tokens_turn2, params, kTurn2Seed, kTurn2Tokens);
  const std::vector<int32_t> turn2_ref =
      RunPlainSampled(ref, full_tokens_turn2, params, kTurn2Seed, kTurn2Tokens);
  if (turn2 != turn2_ref) {
    std::fprintf(stderr,
                 "FAIL: CheckSampledMidRoundStop: turn-2 sampled continuation (post-Reset(), "
                 "post-mid-round-stop) diverges from an independently loaded reference\n");
    return false;
  }
  std::fprintf(stderr,
               "[mtp] CheckSampledMidRoundStop: PrefixState refused the fast path and the turn-2 "
               "sampled continuation matches the reference exactly (%zu tokens)\n",
               kTurn2Tokens);
  return true;
}

std::vector<int32_t> MakeSecondTurnUserTokens(int n) {
  std::vector<int32_t> ids(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) ids[static_cast<size_t>(i)] = 3000 + (i * 67) % 5000;
  return ids;
}

}  // namespace

int main() {
  if (!FileExists(kContainerPath)) {
    return SkipMissing(kContainerPath);
  }

  // Review finding, 2026-09-20: widened from {kBf16, kW4a16} to all four layouts the allmtp
  // container above actually carries -- w4a8/mxfp4 previously had zero coverage from this file.
  const Layout layouts[] = {Layout::kBf16, Layout::kW4a16, Layout::kW4a8, Layout::kMxfp4};
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

    if (!CheckChatMultiTurnMidRoundStop(opts, prompt, MakeSecondTurnUserTokens(12))) {
      std::fprintf(stderr, "FAIL [%s]: CheckChatMultiTurnMidRoundStop\n", LayoutName(layout));
      return 1;
    }
    std::fprintf(stderr, "[PASS] layout=%s CheckChatMultiTurnMidRoundStop\n", LayoutName(layout));

    if (!CheckWideWindowRejectionRewind(opts, prompt, /*wide_k=*/16)) {
      std::fprintf(stderr, "FAIL [%s]: CheckWideWindowRejectionRewind(K=16)\n", LayoutName(layout));
      return 1;
    }
    std::fprintf(stderr, "[PASS] layout=%s CheckWideWindowRejectionRewind(K=16)\n", LayoutName(layout));

    // Milestone 6 stage S2 (docs/sampling.md sections 8-9). Both checks run on every layout -- the
    // same reason the greedy checks above do (a numeric difference between batched verification and
    // single-row decode is layout-dependent, and sampling is more sensitive to it than an argmax).
    if (!CheckSampledSummaryPathExact(opts, prompt, layout)) {
      std::fprintf(stderr, "FAIL [%s]: CheckSampledSummaryPathExact\n", LayoutName(layout));
      return 1;
    }
    std::fprintf(stderr, "[PASS] layout=%s CheckSampledSummaryPathExact\n", LayoutName(layout));

    if (!CheckSampledRoundsMatchPlain(opts, prompt, layout)) {
      std::fprintf(stderr, "FAIL [%s]: CheckSampledRoundsMatchPlain\n", LayoutName(layout));
      return 1;
    }
    std::fprintf(stderr, "[PASS] layout=%s CheckSampledRoundsMatchPlain\n", LayoutName(layout));
    ++ran;
  }

  if (ran == 0) {
    std::fprintf(stderr, "FAIL: no layout ran to completion\n");
    return 1;
  }

  // Stage S2 item 3c: the sampled mid-round-stop bookkeeping. One layout is enough (this is about
  // ProcessMtpRound/PrefixState arithmetic over a sampled round, not about per-layout numerics),
  // and w4a16 is the layout every other real-hardware check in this milestone reports against.
  {
    ModelOptions opts;
    opts.container_path = kContainerPath;
    opts.layout = Layout::kW4a16;
    opts.max_ctx = 256;
    opts.layer_limit = 4;
    if (!CheckSampledMidRoundStop(opts, prompt, MakeSecondTurnUserTokens(12))) {
      std::fprintf(stderr, "FAIL: CheckSampledMidRoundStop\n");
      return 1;
    }
    std::fprintf(stderr, "[PASS] layout=w4a16 CheckSampledMidRoundStop\n");
  }

  // Reduced-vocab draft head (docs/r9700.md R9): separate container (has mtp.draft_head.* tensors),
  // separately SKIPped if missing so a machine that never ran this task's own converter step still
  // runs everything else in this file.
  const char* kDraftVocabContainerPath = "D:/models/r4dx/qwen38-27b-l4-mtp-draftvocab.r4dx";
  if (!FileExists(kDraftVocabContainerPath)) {
    // Loud, grep-able skip (review finding, 2026-09-20): this test still returns 0 (a clean
    // ctest "Passed") on a machine missing this container, so the reduced-vocab draft head --
    // which defaults to ON in production -- would otherwise get zero automated coverage with
    // nothing in a normal `ctest` run (no --output-on-failure/-V) ever showing it happened. Uses
    // the same "[SKIP]" tag test_common.h's SkipMissing() uses for a whole-binary skip, so tooling
    // that already greps ctest logs for that tag catches this partial skip too.
    std::fprintf(stderr,
                 "[SKIP] %s not found -- CheckReducedVocabDraftHeadLossless did NOT run this pass "
                 "(not a product failure; this ctest binary still exits 0 because every OTHER "
                 "check above did run and pass -- see docs/validation.md for how to generate this "
                 "container).\n",
                 kDraftVocabContainerPath);
  } else {
    // kDraftVocabContainerPath is its OWN, separately-converted container -- confirmed (crash
    // reproduced, then fixed here) that it does NOT carry all four of `layouts` above's entries,
    // only the original {bf16, w4a16} pair this loop was written against; reusing the now-widened
    // `layouts` array crashed the process (STATUS_STACK_BUFFER_OVERRUN) attempting w4a8/mxfp4
    // against tensors this container doesn't have (review finding, 2026-09-20 -- widening the MAIN
    // loop's `layouts` must not silently widen this unrelated loop too).
    const Layout draft_vocab_layouts[] = {Layout::kBf16, Layout::kW4a16};
    for (Layout layout : draft_vocab_layouts) {
      if (!CheckReducedVocabDraftHeadLossless(kDraftVocabContainerPath, layout, prompt)) {
        std::fprintf(stderr, "FAIL [%s]: CheckReducedVocabDraftHeadLossless\n", LayoutName(layout));
        return 1;
      }
      std::fprintf(stderr, "[PASS] layout=%s CheckReducedVocabDraftHeadLossless\n", LayoutName(layout));
    }
  }

  return 0;
}
