// tests/model/test_dflash_e2e.cpp -- real end-to-end regression test for DFlash2's own
// "mid-round-stop" bookkeeping (docs/dflash2.md, Milestone 5 stage S3 item 3): a --dflash variant
// of tests/model/test_mtp.cpp::CheckChatMultiTurnMidRoundStop, same technique, same contract.
//
// Model::DecodeStepDflashGreedy atomically commits its ENTIRE returned round vector (minus its own
// last element) into the model's real KV/GDN/DFlash2-ring state in the same call that produced it,
// regardless of whether a caller's own per-token display loop later decides to stop partway through
// that round (--max-tokens reached mid-round). This is EXACTLY the same "committed vs displayed"
// contract MTP's own DecodeStepMtpGreedy already has (mtp_round.hpp's ProcessMtpRound is
// speculation-family-agnostic despite its name -- see that header's file comment), so this test
// drives a real two-turn conversation through a real r4dx::model::Model with real
// DecodeStepDflashGreedy rounds, forces a stop exactly one token short of a round's own boundary,
// and checks (a) PrefixState correctly refuses to treat the resulting turn-2 prompt as a simple
// extension (the desync-detection half of the fix) and (b) a Reset()+re-prefill continuation from
// there matches an independently-loaded sequential (--mtp 0-equivalent, no dflash) reference
// exactly -- the same two assertions CheckChatMultiTurnMidRoundStop makes for MTP.
//
// Needs the REAL 64-layer target container AND the real w4a16 DFlash2 draft container (both ~GB-
// scale, not vendored into the repo) -- SKIPs cleanly (CTest SKIPPED) if either is missing, same
// convention as every other real-hardware-only test in this directory. Only ONE (target, draft)
// layout pair (w4a16/w4a16) is exercised, not the full layout matrix test_mtp.cpp sweeps, to keep
// this test's own real-45GB-container-load cost bounded to a single load per Model construction.
#include <cstdio>
#include <string>
#include <vector>

#include "model.h"
#include "mtp_round.hpp"
#include "prefix_state.h"
#include "test_common.h"

using namespace r4dx_test;
using r4dx::model::Layout;
using r4dx::model::Model;
using r4dx::model::ModelOptions;

namespace {

// Real 64-layer container + real w4a16 DFlash2 draft container (docs/status.md's own paths for
// every other real-hardware perf/e2e pass in this project).
const char* kTargetContainerPath = "D:/models/r4dx/qwen38-27b-v3.r4dx";
const char* kDflashContainerPath = "D:/models/r4dx/qwen38-27b-dflash2-w4a16.r4dx";
constexpr int64_t kDflashK = 7;

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

std::vector<int32_t> MakeSecondTurnUserTokens(int n) {
  std::vector<int32_t> ids(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) ids[static_cast<size_t>(i)] = 3000 + (i * 67) % 5000;
  return ids;
}

bool CheckChatMultiTurnMidRoundStop(const ModelOptions& base_opts,
                                     const std::vector<int32_t>& prompt1,
                                     const std::vector<int32_t>& prompt2_user) {
  ModelOptions dflash_opts = base_opts;
  dflash_opts.dflash_container = kDflashContainerPath;
  dflash_opts.dflash_draft_k = kDflashK;
  auto is_eos = [](int32_t) { return false; };

  // Pass 1 (dry run): replay the natural (unbudgeted) round trajectory to find a round that itself
  // returns >=2 tokens, so stopping display strictly before ITS OWN committed prefix ends is a
  // genuine mid-round stop (not merely landing on the round's own boundary, where displayed would
  // equal committed exactly -- see the derivation below).
  //
  // FOUND while writing this test (out of this test's own scope to fix -- flagged in this stage's
  // open_issues, not touched here): tests/model/test_mtp.cpp's own CheckChatMultiTurnMidRoundStop
  // uses `cumulative + round.size() - 1`, i.e. max_tokens_remaining == round.size()-1 for the
  // target round. ProcessMtpRound's `committed` is UNCONDITIONALLY round[0..size-2] (round.size()-1
  // elements), and with max_tokens_remaining == round.size()-1, `displayed` pushes exactly
  // round[0..size-2] too (the loop's own bound check fires exactly at index round.size()-1, one
  // past the last pushed element) -- i.e. displayed == committed EXACTLY, every time, for ANY
  // round.size() >= 2. `stopped_mid_round` (`committed.size() > displayed.size()`) can therefore
  // MATHEMATICALLY NEVER be true with that formula; the "-1" needs to be "-2" (max_tokens_remaining
  // == round.size()-2, i.e. one token stingier) to leave a genuine 1-token gap. This was not caught
  // before because the 4-layer container that test happens to run against apparently never returns
  // a round with size>=2 within its own first 8 rounds, so it always takes the (correctly handled,
  // non-failing) "cannot force a mid-round stop; skipping" early-return instead of ever reaching the
  // broken arithmetic. This test uses the corrected "-2" formula.
  int64_t stop_after = -1;
  {
    Model probe = Model::Load(dflash_opts);
    int32_t next = Argmax(probe.Prefill(prompt1));
    int64_t cumulative = 0;
    for (int round_idx = 0; round_idx < 8 && stop_after < 0; ++round_idx) {
      std::vector<int32_t> round =
          probe.DecodeStepDflashGreedy(next, kDflashK, /*p_min=*/0.0f, /*n_min=*/0);
      // >=1 required: stop_after==0 would make the real run's own `while (remaining > 0)` loop
      // below never execute a single round at all, which is a degenerate non-test, not a genuine
      // mid-round stop.
      const int64_t candidate = cumulative + static_cast<int64_t>(round.size()) - 2;
      if (round.size() >= 2 && candidate >= 1) stop_after = candidate;
      cumulative += static_cast<int64_t>(round.size());
      next = round.back();
    }
  }
  if (stop_after < 0) {
    std::fprintf(stderr,
                 "[dflash] CheckChatMultiTurnMidRoundStop: no round emitted >=2 tokens in 8 "
                 "rounds on this container/prompt -- cannot force a mid-round stop; skipping "
                 "(not a product failure)\n");
    return true;
  }

  // Turn 1, for real: identical Model construction/prompt/K (greedy + deterministic => reproduces
  // pass 1's own trajectory exactly), this time actually enforcing the max_tokens_remaining budget.
  Model turn = Model::Load(dflash_opts);
  int32_t next = Argmax(turn.Prefill(prompt1));

  r4dx::server::PrefixState prefix;
  prefix.Commit(prompt1, {});  // Prefill() alone already committed prompt1

  std::vector<int32_t> committed_turn1, displayed_turn1;
  int64_t remaining = stop_after;
  bool stopped_mid_round = false;
  while (remaining > 0) {
    std::vector<int32_t> round =
        turn.DecodeStepDflashGreedy(next, kDflashK, /*p_min=*/0.0f, /*n_min=*/0);
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
                 "FAIL: CheckChatMultiTurnMidRoundStop did not land mid-round (stop_after=%lld) "
                 "-- test construction bug, not a product bug\n",
                 static_cast<long long>(stop_after));
    return false;
  }
  prefix.Commit(prompt1, committed_turn1);
  std::fprintf(stderr,
               "[dflash] CheckChatMultiTurnMidRoundStop: turn 1 stopped mid-round: displayed=%zu "
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
               "[dflash] CheckChatMultiTurnMidRoundStop: PrefixState correctly refused the "
               "fast-path extension (forces Reset()+full reprefill, as a real caller must)\n");

  // Recovery path a real caller takes on Extend()==nullopt: Reset() (which also resets THIS
  // Model's own DflashDraft ring -- model.h's Reset() doc comment) + feed the whole turn-2 prompt
  // from scratch, continue with plain (non-dflash) greedy decode -- exactly main()'s "not an
  // extension" branch. Its result must be byte-identical to an INDEPENDENTLY loaded, no-dflash
  // reference Model fed the same turn-2 conversation from scratch.
  turn.Reset();
  prefix.Clear();
  int32_t tok = Argmax(turn.Prefill(full_tokens_turn2));
  constexpr int kTurn2Steps = 6;
  std::vector<int32_t> turn2_sequence;
  for (int i = 0; i < kTurn2Steps; ++i) {
    turn2_sequence.push_back(tok);
    tok = turn.DecodeStepGreedy(tok);
  }

  // `turn` (the dflash-enabled Model, ~1 GiB of drafter on top of the real target's own ~15.5 GiB
  // weights) is no longer needed -- free it BEFORE loading a second full 27B target as `ref`, or
  // both are VRAM-resident at once (this container is real 45 GiB-class weights x2 concepts, and
  // this test's own dry run above already showed Container's own over-commit warning firing when
  // both were briefly alive together; freeing first keeps this test robust on a card with less
  // free VRAM than this development machine's, not just lucky here).
  { Model discard = std::move(turn); }
  Model ref = Model::Load(base_opts);  // no dflash_container -- plain sequential reference
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
               "[dflash] CheckChatMultiTurnMidRoundStop: turn-2 continuation matches the "
               "sequential reference exactly (%d tokens)\n",
               kTurn2Steps);
  return true;
}

// Review finding (2026-09-21, minor): the only lifecycle test above continues turn 2, after
// Reset()+re-prefill, with PLAIN (non-dflash) greedy decode -- nothing exercised Reset() followed by
// CONTINUED DFlash2 decoding, i.e. the drafter's ring restarting from n=0 against a freshly
// re-prefilled target, which is exactly the path a server takes on every prefix-cache miss for a
// session that keeps using --dflash. Checked here by construction rather than by inspection: a
// Model that runs a few real dflash rounds, THEN Reset()s and re-prefills, THEN continues with
// MORE real dflash rounds, must produce a token sequence byte-identical to a second, independently
// loaded Model fed ONLY the post-reset prompt and decoded the same way from a fresh ring -- if
// Reset() left any drafter-ring state behind (a stale InjectedCount(), a stale K/V slot), the two
// would diverge because the "warmed" model's ring would still carry pre-reset positions the fresh
// model never had.
bool CheckResetThenDflashContinuation(const ModelOptions& base_opts,
                                       const std::vector<int32_t>& prompt1,
                                       const std::vector<int32_t>& prompt2) {
  ModelOptions dflash_opts = base_opts;
  dflash_opts.dflash_container = kDflashContainerPath;
  dflash_opts.dflash_draft_k = kDflashK;
  constexpr int kWarmRounds = 3;
  constexpr int kPostResetRounds = 4;

  Model warm = Model::Load(dflash_opts);
  int32_t next = Argmax(warm.Prefill(prompt1));
  for (int i = 0; i < kWarmRounds; ++i) {
    std::vector<int32_t> round = warm.DecodeStepDflashGreedy(next, kDflashK, 0.0f, 0);
    next = round.back();
  }
  warm.Reset();
  next = Argmax(warm.Prefill(prompt2));
  std::vector<int32_t> warmed_sequence;
  for (int i = 0; i < kPostResetRounds; ++i) {
    std::vector<int32_t> round = warm.DecodeStepDflashGreedy(next, kDflashK, 0.0f, 0);
    warmed_sequence.insert(warmed_sequence.end(), round.begin(), round.end());
    next = round.back();
  }
  { Model discard = std::move(warm); }  // free before loading a second full 27B target (see above)

  Model fresh = Model::Load(dflash_opts);
  int32_t fresh_next = Argmax(fresh.Prefill(prompt2));
  std::vector<int32_t> fresh_sequence;
  for (int i = 0; i < kPostResetRounds; ++i) {
    std::vector<int32_t> round = fresh.DecodeStepDflashGreedy(fresh_next, kDflashK, 0.0f, 0);
    fresh_sequence.insert(fresh_sequence.end(), round.begin(), round.end());
    fresh_next = round.back();
  }

  if (warmed_sequence != fresh_sequence) {
    std::fprintf(stderr,
                 "FAIL: CheckResetThenDflashContinuation: post-Reset() DFlash2 decode diverges from "
                 "a freshly-loaded Model given the identical post-reset prompt -- Reset() left "
                 "drafter-ring state behind\n");
    for (size_t i = 0; i < warmed_sequence.size() && i < fresh_sequence.size(); ++i) {
      if (warmed_sequence[i] != fresh_sequence[i]) {
        std::fprintf(stderr, "  first divergence at index %zu: warmed=%d fresh=%d\n", i,
                     warmed_sequence[i], fresh_sequence[i]);
        break;
      }
    }
    return false;
  }
  std::fprintf(stderr,
               "[dflash] CheckResetThenDflashContinuation: warmed-then-reset DFlash2 decode matches "
               "a freshly-loaded Model exactly (%zu tokens)\n",
               warmed_sequence.size());
  return true;
}

}  // namespace

int main() {
  if (!FileExists(kTargetContainerPath)) return SkipMissing(kTargetContainerPath);
  if (!FileExists(kDflashContainerPath)) return SkipMissing(kDflashContainerPath);

  ModelOptions opts;
  opts.container_path = kTargetContainerPath;
  opts.layout = Layout::kW4a16;
  opts.max_ctx = 512;

  const std::vector<int32_t> prompt1 = MakePromptTokens(24);
  const std::vector<int32_t> prompt2_user = MakeSecondTurnUserTokens(12);

  if (!CheckChatMultiTurnMidRoundStop(opts, prompt1, prompt2_user)) {
    std::fprintf(stderr, "FAIL: CheckChatMultiTurnMidRoundStop\n");
    return 1;
  }
  std::fprintf(stderr, "[PASS] CheckChatMultiTurnMidRoundStop (dflash, layout=w4a16)\n");

  if (!CheckResetThenDflashContinuation(opts, prompt1, prompt2_user)) {
    std::fprintf(stderr, "FAIL: CheckResetThenDflashContinuation\n");
    return 1;
  }
  std::fprintf(stderr, "[PASS] CheckResetThenDflashContinuation (dflash, layout=w4a16)\n");
  return 0;
}
