// tests/model/test_mtp_round.cpp -- pure CPU unit test for src/model/mtp_round.hpp's
// ProcessMtpRound, the "what does one MTP round commit vs. display" decision shared by
// src/cli/main.cpp's RunTurn and src/server/engine.cpp's RunRequest (docs/mtp.md's "mid-round"
// gap). No HIP device or model container needed -- plain add_test(), same pattern
// tests/server/test_prefix_state.cpp uses for its own HIP-free half.
//
// This is a regression test for a real bug: an earlier version of both call sites pushed each
// round[ri] into `committed_tokens` from INSIDE the same per-token display loop that could
// `break` early (--max-tokens reached, or an EOS candidate that is not round's own last element),
// which silently dropped every not-yet-visited index even though Model::DecodeStepMtpGreedy had
// already committed it into the model's real KV/GDN state. ProcessMtpRound computes the full
// committed set up front, independent of the display loop, closing that gap.
#include <cstdint>
#include <cstdio>
#include <vector>

#include "mtp_round.hpp"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                           \
  do {                                                                        \
    if (!(cond)) {                                                            \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, \
                   #cond);                                                    \
      ++g_failures;                                                           \
    }                                                                         \
  } while (0)

using r4dx::model::MtpRoundResult;
using r4dx::model::ProcessMtpRound;

bool NeverEos(int32_t) { return false; }

// Every round element (all but the round's own last) must be committed even when the display
// loop breaks on the SECOND element of a 4-element round (a plausible max-tokens boundary) --
// this is exactly the scenario the review finding's suggested test describes.
void TestBreakAtSecondElementStillCommitsRest() {
  const std::vector<int32_t> round = {10, 11, 12, 13};  // size 4: 3 committed, 1 "bonus"/correction
  // max_tokens_remaining=2 forces the display loop to stop after emitting round[0], round[1].
  const MtpRoundResult r = ProcessMtpRound(round, NeverEos, /*max_tokens_remaining=*/2);
  CHECK(r.committed == std::vector<int32_t>({10, 11, 12}));  // ALL of round except its last element
  CHECK(r.displayed == std::vector<int32_t>({10, 11}));
  CHECK(r.hit_max_tokens);
  CHECK(!r.hit_eos);
}

// An EOS candidate that is not the round's own last element must still leave every earlier AND
// later (up to round.size()-2) element in `committed` -- only `displayed` truncates at the EOS.
void TestEosMidRoundStillCommitsWholeRoundMinusLast() {
  const std::vector<int32_t> round = {20, 21, 22, 23};  // 21 is EOS, mid-round
  auto is_eos = [](int32_t t) { return t == 21; };
  const MtpRoundResult r = ProcessMtpRound(round, is_eos, /*max_tokens_remaining=*/100);
  CHECK(r.committed == std::vector<int32_t>({20, 21, 22}));  // round[0..size-2], unconditional
  CHECK(r.displayed == std::vector<int32_t>({20}));           // display stops before the EOS token
  CHECK(r.hit_eos);
  CHECK(!r.hit_max_tokens);
}

// Every draft accepted (no early stop): `displayed` equals the whole round, `committed` is
// everything but the round's own last (bonus/correction) element.
void TestFullAcceptanceDisplaysWholeRound() {
  const std::vector<int32_t> round = {30, 31, 32};
  const MtpRoundResult r = ProcessMtpRound(round, NeverEos, /*max_tokens_remaining=*/100);
  CHECK(r.committed == std::vector<int32_t>({30, 31}));
  CHECK(r.displayed == std::vector<int32_t>({30, 31, 32}));
  CHECK(!r.hit_eos);
  CHECK(!r.hit_max_tokens);
}

// k==0 degenerate round (Model::DecodeStepMtpGreedy's own "single DecodeStepGreedy-equivalent"
// case): round.size()==1, nothing to commit besides the caller's own separately-added seed token.
void TestSingleElementRoundCommitsNothing() {
  const std::vector<int32_t> round = {40};
  const MtpRoundResult r = ProcessMtpRound(round, NeverEos, /*max_tokens_remaining=*/100);
  CHECK(r.committed.empty());
  CHECK(r.displayed == std::vector<int32_t>({40}));
}

// max_tokens_remaining==0: displays nothing, but a 4-element round still commits 3 -- the
// exact "one token of budget left mid-round" case the review finding called out by name.
void TestZeroBudgetStillCommitsWholeRoundMinusLast() {
  const std::vector<int32_t> round = {50, 51, 52, 53};
  const MtpRoundResult r = ProcessMtpRound(round, NeverEos, /*max_tokens_remaining=*/0);
  CHECK(r.committed == std::vector<int32_t>({50, 51, 52}));
  CHECK(r.displayed.empty());
  CHECK(r.hit_max_tokens);
}

void TestEmptyRound() {
  const std::vector<int32_t> round = {};
  const MtpRoundResult r = ProcessMtpRound(round, NeverEos, /*max_tokens_remaining=*/100);
  CHECK(r.committed.empty());
  CHECK(r.displayed.empty());
  CHECK(!r.hit_eos);
  CHECK(!r.hit_max_tokens);
}

}  // namespace

int main() {
  TestBreakAtSecondElementStillCommitsRest();
  TestEosMidRoundStillCommitsWholeRoundMinusLast();
  TestFullAcceptanceDisplaysWholeRound();
  TestSingleElementRoundCommitsNothing();
  TestZeroBudgetStillCommitsWholeRoundMinusLast();
  TestEmptyRound();

  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) FAILED\n", g_failures);
    return 1;
  }
  std::fprintf(stderr, "all checks passed\n");
  return 0;
}
