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
#include "mtp_round.hpp"
#include "prefix_state.h"
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
// conversation through a real r4dx::model::Model with real DecodeStepMtpGreedy rounds, feeding
// their actual output through the SAME r4dx::model::ProcessMtpRound / r4dx::server::PrefixState
// helpers src/cli/main.cpp and src/server/engine.cpp use, with a --max-tokens-equivalent budget
// chosen (by a same-Model, same-seed dry run) to land exactly one token short of a round boundary --
// i.e. the round's own atomic commit genuinely leaves at least one "committed but never displayed"
// token in the model's real state, the exact scenario the mid-round bookkeeping fix exists for.
bool CheckChatMultiTurnMidRoundStop(const ModelOptions& base_opts,
                                     const std::vector<int32_t>& prompt1,
                                     const std::vector<int32_t>& prompt2_user) {
  ModelOptions mtp_opts = base_opts;
  mtp_opts.mtp_draft_k = kDraftK;
  auto is_eos = [](int32_t) { return false; };

  // Pass 1 (dry run): replay the natural (unbudgeted) round trajectory to find a round that itself
  // returns >=2 tokens, so stopping display 1 token short of ITS OWN boundary is a genuine mid-round
  // stop (not merely landing on the previous round's own boundary).
  int64_t stop_after = -1;
  {
    Model probe = Model::Load(mtp_opts);
    int32_t next = Argmax(probe.Prefill(prompt1));
    int64_t cumulative = 0;
    for (int round_idx = 0; round_idx < 8 && stop_after < 0; ++round_idx) {
      std::vector<int32_t> round = probe.DecodeStepMtpGreedy(next, kDraftK);
      if (round.size() >= 2) stop_after = cumulative + static_cast<int64_t>(round.size()) - 1;
      cumulative += static_cast<int64_t>(round.size());
      next = round.back();
    }
  }
  if (stop_after < 0) {
    std::fprintf(stderr,
                 "[mtp] CheckChatMultiTurnMidRoundStop: no round emitted >=2 tokens in 8 rounds on "
                 "this container/prompt -- cannot force a mid-round stop; skipping (not a product "
                 "failure)\n");
    return true;
  }

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
    std::vector<int32_t> round = turn.DecodeStepMtpGreedy(next, kDraftK);
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
    ++ran;
  }

  if (ran == 0) {
    std::fprintf(stderr, "FAIL: no layout ran to completion\n");
    return 1;
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
