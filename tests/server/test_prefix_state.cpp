// tests/server/test_prefix_state.cpp -- pure CPU unit test for src/server/prefix_state.h's
// PrefixState, the prefix-reuse/invalidate-on-failure bookkeeping Engine::RunRequest drives
// (docs/server.md "Prefix reuse", docs/mtp.md's "mid-round" gap). Engine itself is HIP-dependent
// (owns r4dx::model::Model) and cannot be unit-tested on CPU alone (engine.h's own file comment),
// so this is the CPU-testable half of that bookkeeping's contract -- exercised end to end against
// a real Model by tools/server/smoke.ps1 instead.
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

}  // namespace

int main() {
  TestExtendOnEmptyState();
  TestExtendMatchingPrefix();
  TestExtendNonMatchingPrefix();
  TestExtendByteIdenticalRepeatIsNotAnExtension();
  TestExtendShorterPromptIsNotAnExtension();
  TestClearResetsToEmpty();
  TestInvalidateResetsToEmpty();
  TestCommitTracksCommittedNotDisplayedTokens();

  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) FAILED\n", g_failures);
    return 1;
  }
  std::fprintf(stderr, "all checks passed\n");
  return 0;
}
