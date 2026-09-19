// r4dx::server::PrefixState -- the "which tokens are already committed to the Model's KV/GDN
// state" bookkeeping described in docs/server.md's "Prefix reuse" section, pulled out of Engine
// into its own header (pure std::vector<int32_t> arithmetic, no HIP/r4dx::model dependency) so it
// is unit-testable on CPU alone, mirroring src/cli/cli_args.h / src/server/openai_types.h's own
// "split out the HIP-free half" rationale.
//
// MTP-aware Commit() contract (docs/mtp.md's "`--chat` multi-turn + MTP interaction not
// exercised" gap, closed here): a caller must pass the tokens ACTUALLY fed into the model this
// turn (`committed_tokens`), which is not always the same as the tokens shown to the client. An
// MTP round (r4dx::model::Model::DecodeStepMtpGreedy) commits every candidate up to (but not
// including) its own returned vector's LAST element atomically, in one call -- if the caller's own
// per-token loop then stops mid-vector (--max-tokens reached, or an EOS candidate that isn't the
// vector's last element), the model's real state has already advanced past tokens the caller never
// displayed or counted toward the response. Passing only the "displayed" set here would
// under-count `fed_`, and the NEXT request's Extend() could then take the "matches, feed only the
// tail" branch against a model whose real position is ahead of what `fed_` claims -- silently
// misaligning every subsequent token's RoPE position / KV slot. See src/server/engine.cpp's
// RunRequest and src/cli/main.cpp's RunTurn for the two callers that build `committed_tokens`
// correctly.
#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

namespace r4dx::server {

class PrefixState {
 public:
  // Returns the tail of `full_tokens` still needing to be fed if `full_tokens` strictly extends
  // the currently-tracked prefix (fed() is a prefix of full_tokens AND full_tokens is longer);
  // std::nullopt otherwise -- covers both a genuinely different conversation and the "byte-
  // identical repeated request" edge case (full_tokens.size() == fed().size()), both of which the
  // caller should handle the same way: Model::Reset() + Clear() + feed the whole full_tokens.
  std::optional<std::vector<int32_t>> Extend(const std::vector<int32_t>& full_tokens) const {
    if (full_tokens.size() <= fed_.size()) return std::nullopt;
    if (!std::equal(fed_.begin(), fed_.end(), full_tokens.begin())) return std::nullopt;
    return std::vector<int32_t>(full_tokens.begin() + static_cast<ptrdiff_t>(fed_.size()),
                                 full_tokens.end());
  }

  // Call once the model has actually been reset (Model::Reset() or a fresh Load()) -- nothing is
  // fed yet.
  void Clear() { fed_.clear(); }

  // Call after a request completes successfully. `full_tokens`: the prompt this request rendered
  // (already includes everything previously tracked, per Extend()'s contract). `committed_tokens`:
  // this turn's own newly-committed tokens -- see file header comment for why this is not always
  // the same as "the tokens shown to the client".
  void Commit(const std::vector<int32_t>& full_tokens, const std::vector<int32_t>& committed_tokens) {
    fed_ = full_tokens;
    fed_.insert(fed_.end(), committed_tokens.begin(), committed_tokens.end());
  }

  // Call from a catch block around any Model call that may have left the real model state ahead
  // of what fed_ describes (Prefill/DecodeStep*/Reset throwing partway through) -- forces the next
  // request down the full-reset path rather than risking a stale prefix match against a model
  // whose real state has silently diverged.
  void Invalidate() { fed_.clear(); }

  const std::vector<int32_t>& fed() const { return fed_; }

 private:
  std::vector<int32_t> fed_;
};

}  // namespace r4dx::server
