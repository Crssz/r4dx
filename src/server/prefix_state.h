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
//
// PROMPT CHECKPOINT (docs/server.md "Prefix cache"): the GDN recurrent state cannot be rewound, so
// fed() is only reusable when the next prompt re-tokenizes to EVERY committed token, the reply
// included -- and a replayed reply often does not: the chat template trims it (a " yes" reply comes
// back as "yes"), a client drops the reasoning, a tool call re-renders in the template's own shape.
// When the model also saved its state at the end of the prompt (Model::SaveCheckpoint, passed to
// Commit() as `checkpoint_len`), Plan() falls back to that shorter prefix: restore, then feed only
// what follows it. Both candidates obey the same exactness rule -- the new prompt's tokens must
// strictly extend the candidate's, and its image keys must start with the candidate's -- so neither
// can reuse state the new prompt does not describe.
#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

namespace r4dx::server {

// A stable identifier for one image occurrence in a prompt -- its byte content and the grid it
// preprocessed to. See PrefixState::Extend for why token equality alone is not enough once a
// prompt can contain images.
struct ImageKey {
  uint64_t content_hash = 0;  // over the DECODED/encoded image bytes the request supplied
  int64_t grid_t = 0, grid_h = 0, grid_w = 0;
  int64_t token_offset = 0;  // index of this image's first placeholder token in the prompt

  bool operator==(const ImageKey& o) const {
    return content_hash == o.content_hash && grid_t == o.grid_t && grid_h == o.grid_h &&
           grid_w == o.grid_w && token_offset == o.token_offset;
  }
  bool operator!=(const ImageKey& o) const { return !(*this == o); }
};

class PrefixState {
 public:
  // Returns the tail of `full_tokens` still needing to be fed if `full_tokens` strictly extends
  // the currently-tracked prefix (fed() is a prefix of full_tokens AND full_tokens is longer);
  // std::nullopt otherwise -- covers both a genuinely different conversation and the "byte-
  // identical repeated request" edge case (full_tokens.size() == fed().size()), both of which the
  // caller should handle the same way: Model::Reset() + Clear() + feed the whole full_tokens.
  //
  // IMAGES (vision milestone, docs/vision.md "Text-side splicing"): every image-placeholder token
  // is the SAME token id (248056), so two requests carrying two COMPLETELY DIFFERENT pictures at
  // the same position produce byte-identical token sequences. Token equality therefore no longer
  // implies state equality: reusing the prefix would keep turn 1's image in the KV cache while the
  // client believes it sent a new one, and nothing downstream could detect it. `images` is the
  // caller's per-image fingerprint list for THIS request, in prompt order; the prefix is only
  // reusable when this request's images start with exactly the ones already fed. A caller with no
  // images passes an empty vector and gets the original behaviour unchanged.
  //
  // After Invalidate() this refuses every prompt until the next Commit(): an empty fed_ alone
  // cannot tell "the model is at position 0" (fresh Load, or Clear() after a Reset()) from "the
  // model's state is unknown", and the empty prefix would otherwise match anything.
  std::optional<std::vector<int32_t>> Extend(const std::vector<int32_t>& full_tokens,
                                              const std::vector<ImageKey>& images = {}) const {
    if (needs_reset_) return std::nullopt;
    return TailAfter(fed_, full_tokens, images);
  }

  // Engine::RunRequest's decision (see the file comment's PROMPT CHECKPOINT): continue from fed()
  // when Extend() would, else from the checkpointed prompt when the new prompt strictly extends
  // that (same rules, images included), else std::nullopt -- Model::Reset() + Clear() + feed
  // everything, exactly as for Extend().
  struct Reuse {
    bool from_checkpoint = false;  // Model::RestoreCheckpoint() first; the model is then at checkpoint().size()
    std::vector<int32_t> tail;     // then feed these
  };
  std::optional<Reuse> Plan(const std::vector<int32_t>& full_tokens,
                            const std::vector<ImageKey>& images = {}) const {
    if (needs_reset_) return std::nullopt;
    if (auto tail = TailAfter(fed_, full_tokens, images)) return Reuse{false, std::move(*tail)};
    if (!has_checkpoint_) return std::nullopt;
    if (auto tail = TailAfter(checkpoint_, full_tokens, images)) return Reuse{true, std::move(*tail)};
    return std::nullopt;
  }

  // Call once the model has actually been reset (Model::Reset() or a fresh Load()) -- nothing is
  // fed yet, and Model::Reset() drops the model's checkpoint too. Does not lift a pending
  // Invalidate(); only Commit() does, once a request has run to completion on the reset model.
  void Clear() {
    fed_.clear();
    fed_images_.clear();
    DropCheckpoint();
  }

  // Call after a request completes successfully. `full_tokens`: the prompt this request rendered
  // (already includes everything previously tracked, per Extend()'s contract). `committed_tokens`:
  // this turn's own newly-committed tokens -- see file header comment for why this is not always
  // the same as "the tokens shown to the client".
  // `images`: this request's own image fingerprints, in prompt order -- see Extend().
  // `checkpoint_len`: the model saved its state right after prefilling the first `*checkpoint_len`
  // tokens of `full_tokens` (Model::SaveCheckpoint) -- the prompt's end, or just before it (engine.cpp:
  // a thinking prompt's trailing "\n") -- replacing any older checkpoint; std::nullopt drops the
  // recorded one. Every image must lie inside those tokens (fed_images_ serves both prefixes).
  void Commit(const std::vector<int32_t>& full_tokens, const std::vector<int32_t>& committed_tokens,
              const std::vector<ImageKey>& images = {},
              std::optional<size_t> checkpoint_len = std::nullopt) {
    fed_ = full_tokens;
    fed_.insert(fed_.end(), committed_tokens.begin(), committed_tokens.end());
    fed_images_ = images;
    needs_reset_ = false;
    if (checkpoint_len && *checkpoint_len > 0 && *checkpoint_len <= full_tokens.size()) {
      checkpoint_.assign(full_tokens.begin(), full_tokens.begin() + static_cast<ptrdiff_t>(*checkpoint_len));
      has_checkpoint_ = true;
    } else {
      DropCheckpoint();
    }
  }

  // Call from a catch block around any Model call that may have left the real model state ahead
  // of what fed_ describes (Prefill/DecodeStep*/Reset throwing partway through) -- forces the next
  // request down the full-reset path rather than risking a stale prefix match against a model
  // whose real state has silently diverged. Clearing fed_ alone would not: Extend() treats an
  // empty fed_ as a model at position 0 and would hand back the whole prompt as the "tail". The
  // checkpoint goes too: the next request resets, and the Reset() drops the model's copy.
  void Invalidate() {
    fed_.clear();
    fed_images_.clear();
    DropCheckpoint();
    needs_reset_ = true;
  }

  const std::vector<int32_t>& fed() const { return fed_; }
  const std::vector<ImageKey>& fed_images() const { return fed_images_; }
  bool has_checkpoint() const { return has_checkpoint_; }
  const std::vector<int32_t>& checkpoint() const { return checkpoint_; }

 private:
  // The tail of `full_tokens` after `prefix`, if `full_tokens` strictly extends it and `images`
  // strictly extend fed_images_ in the sense Extend() documents. fed_images_ serves both prefixes:
  // every image lies inside a prompt, and the checkpoint IS the last committed prompt.
  std::optional<std::vector<int32_t>> TailAfter(const std::vector<int32_t>& prefix,
                                                const std::vector<int32_t>& full_tokens,
                                                const std::vector<ImageKey>& images) const {
    if (full_tokens.size() <= prefix.size()) return std::nullopt;
    if (!std::equal(prefix.begin(), prefix.end(), full_tokens.begin())) return std::nullopt;
    if (images.size() < fed_images_.size()) return std::nullopt;
    if (!std::equal(fed_images_.begin(), fed_images_.end(), images.begin())) return std::nullopt;
    // An image that begins inside the already-fed prefix but was not recorded as fed would mean
    // the caller's own bookkeeping disagrees with this one; refuse rather than guess.
    for (size_t i = fed_images_.size(); i < images.size(); ++i) {
      if (images[i].token_offset < static_cast<int64_t>(prefix.size())) return std::nullopt;
    }
    return std::vector<int32_t>(full_tokens.begin() + static_cast<ptrdiff_t>(prefix.size()),
                                 full_tokens.end());
  }
  void DropCheckpoint() {
    checkpoint_.clear();
    has_checkpoint_ = false;
  }

  std::vector<int32_t> fed_;
  std::vector<ImageKey> fed_images_;
  bool needs_reset_ = false;  // set by Invalidate(), cleared by Commit()
  std::vector<int32_t> checkpoint_;  // the tokens the model's checkpoint was saved after
  bool has_checkpoint_ = false;
};

}  // namespace r4dx::server
