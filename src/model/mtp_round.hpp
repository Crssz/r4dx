// r4dx::model::ProcessMtpRound -- pulls the "what does one MTP round commit vs. display" decision
// out of src/cli/main.cpp's RunTurn and src/server/engine.cpp's RunRequest into one pure, HIP-free,
// unit-testable function (mirrors src/server/prefix_state.h's own "split out the HIP-free half"
// rationale), because both call sites need to make EXACTLY the same decision and getting it wrong
// is the bug this file closes.
//
// Model::DecodeStepMtpGreedy (model.h) atomically commits its ENTIRE returned `round` vector minus
// its own last element into the model's real KV/GDN state, in the same call that produced it --
// regardless of whether a caller's own per-token display loop later decides to stop partway
// through `round` (--max-tokens reached, or an EOS candidate that is not round's own last element).
// A caller that only records the tokens its display loop actually visited under-counts what the
// model's real state holds; the next turn's prefix-reuse fast path then desyncs silently (see
// src/server/prefix_state.h's file comment for the exact failure mode this caused before this fix
// closed it -- docs/mtp.md's "mid-round" gap).
//
// ProcessMtpRound separates the two questions PROCESS_MTP_ROUND ONCE, up front, before any display
// side effect (stdout printing, SSE token emission, ...) runs, so a `break` in the caller's own
// display loop can never lose a committed token:
//   - committed: ALL of round[0 .. round.size()-2] (round.size()-1 tokens, or empty if
//     round.size()<=1) -- unconditional, exactly what DecodeStepMtpGreedy already put in the
//     model's state. The caller must still separately add its own `token_id` seed (the value
//     DecodeStepMtpGreedy was called with) -- ProcessMtpRound only covers `round` itself.
//   - displayed: the prefix of `round` the caller should actually show/emit this round, applying
//     is_eos and max_tokens_remaining in round order and stopping at the first one that fires.
#pragma once

#include <cstdint>
#include <vector>

namespace r4dx::model {

struct MtpRoundResult {
  std::vector<int32_t> committed;  // see file comment -- append to "already fed" bookkeeping
  std::vector<int32_t> displayed;  // tokens to actually show/emit, in order
  bool hit_eos = false;            // true iff `displayed`'s stop was an EOS candidate
  bool hit_max_tokens = false;     // true iff `displayed`'s stop was max_tokens_remaining running out
};

// is_eos(int32_t) -> bool. max_tokens_remaining: how many more tokens the caller is willing to
// display this round (>=0; 0 is valid and immediately stops with an empty `displayed`).
template <typename IsEos>
MtpRoundResult ProcessMtpRound(const std::vector<int32_t>& round, IsEos&& is_eos,
                                int64_t max_tokens_remaining) {
  MtpRoundResult r;
  if (round.size() > 1) {
    r.committed.assign(round.begin(), round.end() - 1);
  }
  for (int32_t tok : round) {
    if (is_eos(tok)) {
      r.hit_eos = true;
      break;
    }
    if (static_cast<int64_t>(r.displayed.size()) >= max_tokens_remaining) {
      r.hit_max_tokens = true;
      break;
    }
    r.displayed.push_back(tok);
  }
  return r;
}

}  // namespace r4dx::model
