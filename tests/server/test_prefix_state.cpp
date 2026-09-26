// tests/server/test_prefix_state.cpp -- pure CPU unit test for src/server/prefix_state.h's
// PrefixState, the prefix-reuse/invalidate-on-failure/prompt-checkpoint bookkeeping
// Engine::RunRequest drives (docs/server.md "Prefix reuse" and "Prefix cache", docs/mtp.md's
// "mid-round" gap). Engine itself is HIP-dependent
// (owns r4dx::model::Model) and cannot be unit-tested on CPU alone (engine.h's own file comment),
// so this is the CPU-testable half of that bookkeeping's contract -- exercised end to end against
// a real Model by tools/server/smoke.ps1 instead.
#include <algorithm>
#include <cstdio>
#include <optional>
#include <vector>

#include "prefix_state.h"

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

using r4dx::server::PrefixState;

void TestExtendOnEmptyState() {
  PrefixState p;
  // Nothing fed yet -- any non-empty prompt "extends" the empty prefix (the whole thing is new).
  auto tail = p.Extend({1, 2, 3});
  CHECK(tail.has_value());
  CHECK(*tail == std::vector<int32_t>({1, 2, 3}));
}

void TestExtendMatchingPrefix() {
  PrefixState p;
  p.Commit(/*full_tokens=*/{1, 2, 3}, /*committed_tokens=*/{7, 8});
  CHECK(p.fed() == std::vector<int32_t>({1, 2, 3, 7, 8}));

  auto tail = p.Extend({1, 2, 3, 7, 8, 9, 10});
  CHECK(tail.has_value());
  CHECK(*tail == std::vector<int32_t>({9, 10}));
}

void TestExtendNonMatchingPrefix() {
  PrefixState p;
  p.Commit({1, 2, 3}, {7, 8});
  // Diverges at index 1 (2 vs 99) -- not an extension.
  auto tail = p.Extend({1, 99, 3, 7, 8, 9});
  CHECK(!tail.has_value());
}

void TestExtendByteIdenticalRepeatIsNotAnExtension() {
  PrefixState p;
  p.Commit({1, 2, 3}, {7, 8});
  // full_tokens.size() == fed().size() exactly: nothing new to feed -- Extend() must report "not
  // an extension" (caller degrades to a full reset+re-prefill) rather than returning an empty
  // tail Model::Prefill would throw on (docs/server.md's own "byte-identical repeated request"
  // note, carried over from the pre-PrefixState code this replaces).
  auto tail = p.Extend({1, 2, 3, 7, 8});
  CHECK(!tail.has_value());
}

void TestExtendShorterPromptIsNotAnExtension() {
  PrefixState p;
  p.Commit({1, 2, 3, 4, 5}, {});
  auto tail = p.Extend({1, 2, 3});
  CHECK(!tail.has_value());
}

void TestClearResetsToEmpty() {
  PrefixState p;
  p.Commit({1, 2, 3}, {7, 8});
  p.Clear();
  CHECK(p.fed().empty());
  auto tail = p.Extend({9, 10});
  CHECK(tail.has_value());
  CHECK(*tail == std::vector<int32_t>({9, 10}));
}

void TestInvalidateResetsToEmpty() {
  PrefixState p;
  p.Commit({1, 2, 3}, {7, 8});
  p.Invalidate();
  CHECK(p.fed().empty());
}

// Regression: Invalidate() used to only clear fed_, and Extend() treats an empty fed_ as "model at
// position 0" -- so after a request threw mid-Prefill/Decode, the NEXT request extended the empty
// prefix, skipped Engine's Model::Reset(), and was fed on top of the failed request's leftover
// KV/GDN state. Every prompt must be refused until a request has run to Commit() again.
void TestInvalidateForcesResetUntilCommit() {
  PrefixState p;
  p.Commit({1, 2, 3}, {7, 8});
  p.Invalidate();
  CHECK(!p.Extend({1, 2, 3, 7, 8, 9}).has_value());  // extends the old prefix
  CHECK(!p.Extend({42, 43}).has_value());            // unrelated conversation
  CHECK(!p.Extend({1}).has_value());

  // Engine's recovery: Extend()==nullopt -> Model::Reset() + Clear() + feed the whole prompt. The
  // flag survives Clear() (the request can still fail after the reset) and lifts only on Commit().
  p.Clear();
  CHECK(!p.Extend({42, 43}).has_value());
  p.Commit({42, 43}, {44});
  auto tail = p.Extend({42, 43, 44, 45});
  CHECK(tail.has_value());
  CHECK(*tail == std::vector<int32_t>({45}));

  // A second failure after recovery re-arms it.
  p.Invalidate();
  CHECK(!p.Extend({42, 43, 44, 45}).has_value());

  // Images take the same path.
  using r4dx::server::ImageKey;
  const ImageKey a{/*content_hash=*/0xAAAA, 1, 28, 28, /*token_offset=*/1};
  PrefixState q;
  q.Commit({1, 248056, 248056, 9}, {}, {a});
  q.Invalidate();
  CHECK(!q.Extend({1, 248056, 248056, 9, 10}, {a}).has_value());
}

// docs/mtp.md's "mid-round" gap: an MTP round that stops mid-vector still committed every token
// up to (not including) its own last element -- Commit()'s `committed_tokens` argument is exactly
// that superset, distinct from whatever subset was actually shown to the client. This test checks
// PrefixState.fed() reflects the FULL committed set, not just a caller-supplied "displayed" one.
void TestCommitTracksCommittedNotDisplayedTokens() {
  PrefixState p;
  const std::vector<int32_t> full_tokens = {1, 2, 3};       // e.g. this turn's whole prompt
  // Simulates an MTP round [d0, d1(EOS), corrected] that stopped at d1: d0 and d1 were both
  // already committed atomically by the round call (only `corrected`, the round's last element,
  // was not), even though only d0 was ever displayed to the client.
  const std::vector<int32_t> committed_this_turn = {100 /*seed*/, 101 /*d0*/, 102 /*d1, EOS*/};
  p.Commit(full_tokens, committed_this_turn);
  CHECK(p.fed() == std::vector<int32_t>({1, 2, 3, 100, 101, 102}));

  // The NEXT turn's full_tokens (chat-template re-render) only include what was actually
  // DISPLAYED (100, 101 -- not the withheld EOS candidate 102), so it must NOT be treated as an
  // extension of the true committed state -- this is exactly the divergence PrefixState exists to
  // catch (forcing a Reset()+re-prefill instead of silently misaligned positions).
  auto tail = p.Extend({1, 2, 3, 100, 101, 200});
  CHECK(!tail.has_value());
}

// Vision milestone (docs/vision.md "Text-side splicing"): every image placeholder is the SAME
// token id, so two requests carrying two DIFFERENT pictures at the same position tokenize
// identically. These cover the four cases that distinction creates.
void TestImageAwarePrefixReuse() {
  using r4dx::server::ImageKey;
  const ImageKey a{/*content_hash=*/0xAAAA, 1, 28, 28, /*token_offset=*/3};
  ImageKey b = a;
  b.content_hash = 0xBBBB;  // a DIFFERENT picture, same grid, same place in the prompt
  const std::vector<int32_t> turn1 = {1, 2, 3, 248056, 248056, 9};
  const std::vector<int32_t> turn2 = {1, 2, 3, 248056, 248056, 9, 50, 51, 52};

  // Same image, extended conversation: reuse, exactly as for text.
  {
    PrefixState p;
    p.Commit(turn1, {}, {a});
    auto tail = p.Extend(turn2, {a});
    CHECK(tail.has_value());
    CHECK(*tail == std::vector<int32_t>({50, 51, 52}));
  }
  // Byte-identical tokens, DIFFERENT picture: must NOT reuse. This is the case that exists only
  // because of images -- the token check alone passes.
  {
    PrefixState p;
    p.Commit(turn1, {}, {a});
    CHECK(std::equal(turn1.begin(), turn1.end(), turn2.begin()));  // the tokens really do match
    CHECK(!p.Extend(turn2, {b}).has_value());
  }
  // The client dropped the image from the re-rendered conversation: also not an extension.
  {
    PrefixState p;
    p.Commit(turn1, {}, {a});
    CHECK(!p.Extend(turn2, {}).has_value());
  }
  // A NEW image appears in the new tail (offset past what was fed): still a valid extension.
  {
    PrefixState p;
    p.Commit(turn1, {}, {a});
    ImageKey c{0xCCCC, 1, 26, 38, /*token_offset=*/7};
    auto tail = p.Extend(turn2, {a, c});
    CHECK(tail.has_value());
  }
  // ...but a new image claiming an offset INSIDE the already-fed prefix is caller-bookkeeping
  // disagreement, not a reusable prefix.
  {
    PrefixState p;
    p.Commit(turn1, {}, {a});
    ImageKey c{0xCCCC, 1, 26, 38, /*token_offset=*/1};
    CHECK(!p.Extend(turn2, {a, c}).has_value());
  }
  // Clear()/Invalidate() drop the image list too, or the next request would compare against a
  // conversation that no longer exists.
  {
    PrefixState p;
    p.Commit(turn1, {}, {a});
    p.Clear();
    CHECK(p.fed_images().empty());
    p.Commit(turn1, {}, {a});
    p.Invalidate();
    CHECK(p.fed_images().empty());
  }
}

// ---- prompt checkpoint (prefix_state.h's PROMPT CHECKPOINT, docs/server.md "Prefix cache") ------
// Symbolic ids for the 2026-09-26 failure: turn 1's prompt ends with the generation prompt's
// "</think>\n\n"; the model answers " yes" (leading space) then EOS; the client replays it and the
// chat template's `|trim` renders "yes" -- a different token.
constexpr int32_t kSpaceYes = 1000;  // " yes", what was generated and committed
constexpr int32_t kYes = 2000;       // "yes", what the re-render contains
constexpr int32_t kImEnd = 3000;
const std::vector<int32_t> kTurn1 = {1, 2, 3, 4};                               // ...assistant\n<think>\n\n</think>\n\n
const std::vector<int32_t> kTurn2 = {1, 2, 3, 4, kYes, kImEnd, 50, 51, 52, 53};  // + "yes<|im_end|>\n<|im_start|>user..."

void TestCheckpointRecoversTrimmedReply() {
  // Without a checkpoint this is the bug: fed() = turn 1 + " yes", which turn 2 does not extend, so
  // the only option is a full re-prefill.
  {
    PrefixState p;
    p.Commit(kTurn1, {kSpaceYes});
    CHECK(!p.Extend(kTurn2).has_value());
    CHECK(!p.Plan(kTurn2).has_value());
    CHECK(!p.has_checkpoint());
  }
  // With one, turn 2 restores the state after turn 1's prompt and feeds only what follows it --
  // the re-rendered reply included, which is what makes the result exact.
  {
    PrefixState p;
    p.Commit(kTurn1, {kSpaceYes}, {}, /*checkpoint_len=*/kTurn1.size());
    CHECK(p.has_checkpoint() && p.checkpoint() == kTurn1);
    CHECK(!p.Extend(kTurn2).has_value());  // Extend() is fed()-only, unchanged
    auto plan = p.Plan(kTurn2);
    CHECK(plan.has_value());
    CHECK(plan->from_checkpoint);
    CHECK(plan->tail == std::vector<int32_t>(kTurn2.begin() + 4, kTurn2.end()));
  }
}

// A reply that DOES re-tokenize keeps the longer reuse: fed() wins over the checkpoint.
void TestCheckpointPrefersFullSequence() {
  PrefixState p;
  p.Commit(kTurn1, {kYes}, {}, /*checkpoint_len=*/kTurn1.size());
  auto plan = p.Plan(kTurn2);
  CHECK(plan.has_value());
  CHECK(!plan->from_checkpoint);
  CHECK(plan->tail == std::vector<int32_t>(kTurn2.begin() + 5, kTurn2.end()));
  // An empty reply (EOS first, nothing committed): fed() == the checkpoint, the same tail either way.
  PrefixState q;
  q.Commit(kTurn1, {}, {}, /*checkpoint_len=*/kTurn1.size());
  plan = q.Plan(kTurn2);
  CHECK(plan.has_value() && !plan->from_checkpoint && plan->tail.size() == kTurn2.size() - 4);
}

// The checkpoint is exact too: anything that does not strictly extend the checkpointed prompt resets.
void TestCheckpointNeverReusesADifferentPrompt() {
  PrefixState p;
  p.Commit(kTurn1, {kSpaceYes}, {}, /*checkpoint_len=*/kTurn1.size());
  CHECK(!p.Plan({1, 2, 9, 4, kYes, kImEnd}).has_value());  // diverges inside the prompt
  CHECK(!p.Plan(kTurn1).has_value());                       // byte-identical: nothing left to feed
  CHECK(!p.Plan({1, 2, 3}).has_value());                    // shorter
  CHECK(!p.Plan({7, 8}).has_value());                       // another conversation
}

// Vision: the image-key rules hold for the checkpoint exactly as for fed() -- the case the task
// names: a different picture at the same position must NOT reuse, even though every placeholder is
// the same token id and the reply mismatch sends the decision to the checkpoint.
void TestCheckpointImageRules() {
  using r4dx::server::ImageKey;
  const ImageKey a{/*content_hash=*/0xAAAA, 1, 28, 28, /*token_offset=*/1};
  ImageKey b = a;
  b.content_hash = 0xBBBB;
  const std::vector<int32_t> turn1 = {1, 248056, 248056, 9, 4};
  const std::vector<int32_t> turn2 = {1, 248056, 248056, 9, 4, kYes, kImEnd, 50, 51};
  PrefixState p;
  p.Commit(turn1, {kSpaceYes}, {a}, /*checkpoint_len=*/turn1.size());

  auto same = p.Plan(turn2, {a});
  CHECK(same.has_value() && same->from_checkpoint);
  CHECK(same.has_value() && same->tail == std::vector<int32_t>({kYes, kImEnd, 50, 51}));
  CHECK(!p.Plan(turn2, {b}).has_value());  // different image, same tokens
  CHECK(!p.Plan(turn2, {}).has_value());   // image dropped from the replay
  // A new image in the new tail (past the checkpoint) is fine; one claiming an offset inside the
  // checkpointed prompt is bookkeeping disagreement.
  ImageKey c{0xCCCC, 1, 26, 38, /*token_offset=*/7};
  CHECK(p.Plan(turn2, {a, c}).has_value());
  c.token_offset = 3;
  CHECK(!p.Plan(turn2, {a, c}).has_value());
}

// Thinking on: the prompt ends "<think>" "\n", and a client that drops the reasoning replays
// "<think>" "\n\n" "</think>" "\n\n" ... -- "\n\n" is one token, so the replay diverges AT the prompt's
// last token. engine.cpp therefore checkpoints one token early; the record is that shorter prefix.
void TestCheckpointBeforeThePromptEnd() {
  constexpr int32_t kThink = 500, kNl = 198, kNlNl = 271, kThinkEnd = 501;
  const std::vector<int32_t> turn1 = {1, 2, kThink, kNl};
  const std::vector<int32_t> turn2 = {1, 2, kThink, kNlNl, kThinkEnd, kNlNl, kYes, kImEnd, 50, 51};
  {
    PrefixState p;  // checkpoint at the prompt's end: never a prefix of the replay
    p.Commit(turn1, {77, 78}, {}, turn1.size());
    CHECK(!p.Plan(turn2).has_value());
  }
  PrefixState p;
  p.Commit(turn1, {77, 78}, {}, turn1.size() - 1);
  CHECK(p.checkpoint() == std::vector<int32_t>({1, 2, kThink}));
  auto plan = p.Plan(turn2);
  CHECK(plan.has_value() && plan->from_checkpoint);
  CHECK(plan.has_value() && plan->tail == std::vector<int32_t>(turn2.begin() + 3, turn2.end()));
  // fed() still covers the whole committed sequence, and wins when the replay does reproduce it.
  std::vector<int32_t> full = turn1;
  full.insert(full.end(), {77, 78, 60});
  plan = p.Plan(full);
  CHECK(plan.has_value() && !plan->from_checkpoint && plan->tail == std::vector<int32_t>({60}));
  // A length past the prompt, or zero, is not a checkpoint.
  p.Commit(turn1, {}, {}, turn1.size() + 1);
  CHECK(!p.has_checkpoint());
  p.Commit(turn1, {}, {}, 0);
  CHECK(!p.has_checkpoint());
}

// Lifetime: each Commit() replaces the checkpoint (or drops it), and Clear()/Invalidate() drop it --
// they pair with Model::Reset(), which drops the model's copy.
void TestCheckpointLifetime() {
  // Turn 3 extends turn 2's prompt, not turn 1's: the checkpoint moved.
  {
    PrefixState p;
    p.Commit(kTurn1, {kSpaceYes}, {}, kTurn1.size());
    p.Commit(kTurn2, {kSpaceYes}, {}, kTurn2.size());
    std::vector<int32_t> turn3 = kTurn2;
    turn3.insert(turn3.end(), {kYes, kImEnd, 60});
    auto plan = p.Plan(turn3);
    CHECK(plan.has_value() && plan->from_checkpoint && plan->tail.size() == 3);
    CHECK(p.checkpoint() == kTurn2);
  }
  // A request that did not checkpoint drops the old one (the model's copy no longer matches).
  {
    PrefixState p;
    p.Commit(kTurn1, {kSpaceYes}, {}, kTurn1.size());
    p.Commit(kTurn1, {kSpaceYes}, {}, std::nullopt);
    CHECK(!p.has_checkpoint() && !p.Plan(kTurn2).has_value());
  }
  {
    PrefixState p;
    p.Commit(kTurn1, {kSpaceYes}, {}, kTurn1.size());
    p.Clear();
    CHECK(!p.has_checkpoint() && p.checkpoint().empty());
    CHECK(p.Plan(kTurn2).has_value() && !p.Plan(kTurn2)->from_checkpoint);  // empty fed(): whole prompt
  }
  // After a failed request nothing is reused, the checkpoint included, until the next Commit().
  {
    PrefixState p;
    p.Commit(kTurn1, {kSpaceYes}, {}, kTurn1.size());
    p.Invalidate();
    CHECK(!p.has_checkpoint());
    CHECK(!p.Plan(kTurn2).has_value());
    p.Clear();
    CHECK(!p.Plan(kTurn2).has_value());
    p.Commit(kTurn2, {}, {}, kTurn2.size());
    CHECK(p.Plan({1, 2, 3, 4, kYes, kImEnd, 50, 51, 52, 53, 70}).has_value());
  }
}

}  // namespace

int main() {
  TestExtendOnEmptyState();
  TestExtendMatchingPrefix();
  TestExtendNonMatchingPrefix();
  TestExtendByteIdenticalRepeatIsNotAnExtension();
  TestExtendShorterPromptIsNotAnExtension();
  TestClearResetsToEmpty();
  TestInvalidateResetsToEmpty();
  TestInvalidateForcesResetUntilCommit();
  TestCommitTracksCommittedNotDisplayedTokens();
  TestImageAwarePrefixReuse();
  TestCheckpointRecoversTrimmedReply();
  TestCheckpointPrefersFullSequence();
  TestCheckpointNeverReusesADifferentPrompt();
  TestCheckpointImageRules();
  TestCheckpointBeforeThePromptEnd();
  TestCheckpointLifetime();

  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) FAILED\n", g_failures);
    return 1;
  }
  std::fprintf(stderr, "all checks passed\n");
  return 0;
}
