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
  std::optional<std::vector<int32_t>> Extend(const std::vector<int32_t>& full_tokens,
                                              const std::vector<ImageKey>& images = {}) const {
    if (full_tokens.size() <= fed_.size()) return std::nullopt;
    if (!std::equal(fed_.begin(), fed_.end(), full_tokens.begin())) return std::nullopt;
    if (images.size() < fed_images_.size()) return std::nullopt;
    if (!std::equal(fed_images_.begin(), fed_images_.end(), images.begin())) return std::nullopt;
    // An image that begins inside the already-fed prefix but was not recorded as fed would mean
    // the caller's own bookkeeping disagrees with this one; refuse rather than guess.
    for (size_t i = fed_images_.size(); i < images.size(); ++i) {
      if (images[i].token_offset < static_cast<int64_t>(fed_.size())) return std::nullopt;
    }
    return std::vector<int32_t>(full_tokens.begin() + static_cast<ptrdiff_t>(fed_.size()),
                                 full_tokens.end());
  }

  // Call once the model has actually been reset (Model::Reset() or a fresh Load()) -- nothing is
  // fed yet.
  void Clear() {
    fed_.clear();
    fed_images_.clear();
  }

  // Call after a request completes successfully. `full_tokens`: the prompt this request rendered
  // (already includes everything previously tracked, per Extend()'s contract). `committed_tokens`:
  // this turn's own newly-committed tokens -- see file header comment for why this is not always
  // the same as "the tokens shown to the client".
  // `images`: this request's own image fingerprints, in prompt order -- see Extend().
  void Commit(const std::vector<int32_t>& full_tokens, const std::vector<int32_t>& committed_tokens,
              const std::vector<ImageKey>& images = {}) {
    fed_ = full_tokens;
    fed_.insert(fed_.end(), committed_tokens.begin(), committed_tokens.end());
    fed_images_ = images;
  }

  // Call from a catch block around any Model call that may have left the real model state ahead
  // of what fed_ describes (Prefill/DecodeStep*/Reset throwing partway through) -- forces the next
  // request down the full-reset path rather than risking a stale prefix match against a model
  // whose real state has silently diverged.
  void Invalidate() {
    fed_.clear();
    fed_images_.clear();
  }

  const std::vector<int32_t>& fed() const { return fed_; }
  const std::vector<ImageKey>& fed_images() const { return fed_images_; }

 private:
  std::vector<int32_t> fed_;
  std::vector<ImageKey> fed_images_;
};

}  // namespace r4dx::server
