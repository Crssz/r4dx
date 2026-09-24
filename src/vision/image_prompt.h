// src/vision/image_prompt.h -- the ONE placeholder-expansion routine every user-facing entry point
// (r4dx-cli's --image, r4dx-server's image_url content parts, tests/vision/tool_vision_chat's own
// driver) needs: turning a chat-template-rendered token sequence's single `<|image_pad|>` marker
// per image into that image's real MERGED-token-count run of placeholders, and recording where each
// run landed so the caller can build a Model::ImageSpan for it (docs/vision.md "Text-side
// splicing").
//
// Deliberately does NOT depend on r4dx::model::Model -- src/vision has no r4dx_model dependency
// (CMakeLists.txt's own file comment), so ImagePlaceholderSpan mirrors Model::ImageSpan's fields
// structurally rather than reusing that type; every caller that already includes model.h converts
// the two with a trivial field-for-field copy (see src/cli/main.cpp / src/server/engine.cpp).
#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "preprocess.h"  // GridThw

namespace r4dx::vision {

// One image's placeholder run, once expanded. `embeds` is a device pointer (this image's own
// [tokens, hidden] bf16 rows, e.g. one slice of Model::EncodeImages' output) -- left null by a
// caller that only needs the offsets/grid (e.g. to size a prefix-cache ImageKey before deciding
// which images actually need re-encoding).
struct ImagePlaceholderSpan {
  int64_t offset = 0;  // filled in by ExpandImagePlaceholders; ignored on input
  int64_t tokens = 0;  // filled in by ExpandImagePlaceholders; ignored on input
  GridThw grid;
  const uint16_t* embeds = nullptr;
  // docs/tp.md 8.3: true => `embeds` is host memory (a tensor-parallel ImageRows), carried through
  // to Model::ImageSpan::embeds_on_host. False at TP=1 (device rows) -- and everywhere until P5.
  bool embeds_on_host = false;
};

struct ExpandedImagePrompt {
  std::vector<int32_t> tokens;
  std::vector<ImagePlaceholderSpan> spans;
};

// `raw_tokens`: the chat-template-rendered, already-tokenized prompt, carrying exactly one
// `image_token_id` per image content part the template saw (Qwen3.8-27B's own
// `<|vision_start|><|image_pad|><|vision_end|>` convention -- one `<|image_pad|>` token per image).
// `images`: this prompt's images, IN THE SAME ORDER their content parts appeared to the template
// (only `.grid` and `.embeds` are read; `.offset`/`.tokens` are overwritten). Expands each single
// placeholder into `grid.MergedTokenCount(merge_size)` copies of the same token id and returns the
// per-image span, with `offset` relative to the EXPANDED token vector (index 0 == the first token
// of `raw_tokens`, matching Model::PrefillMultimodal's own `ImageSpan::offset` contract for a call
// whose `token_ids` starts at `raw_tokens[0]`).
//
// Throws std::runtime_error if the placeholder count in `raw_tokens` does not exactly match
// `images.size()` -- a caller-bookkeeping mismatch (a rendered prompt that doesn't agree with how
// many images were actually decoded) that must never silently splice the wrong image's rows.
inline ExpandedImagePrompt ExpandImagePlaceholders(const std::vector<int32_t>& raw_tokens,
                                                    int32_t image_token_id,
                                                    const std::vector<ImagePlaceholderSpan>& images,
                                                    int merge_size) {
  ExpandedImagePrompt out;
  out.tokens.reserve(raw_tokens.size());
  size_t next_image = 0;
  for (const int32_t id : raw_tokens) {
    if (id != image_token_id) {
      out.tokens.push_back(id);
      continue;
    }
    if (next_image >= images.size()) {
      throw std::runtime_error(
          "rendered prompt has more image placeholders than images were supplied");
    }
    ImagePlaceholderSpan sp = images[next_image];
    sp.offset = static_cast<int64_t>(out.tokens.size());
    sp.tokens = sp.grid.MergedTokenCount(merge_size);
    out.spans.push_back(sp);
    out.tokens.insert(out.tokens.end(), static_cast<size_t>(sp.tokens), image_token_id);
    ++next_image;
  }
  if (next_image != images.size()) {
    throw std::runtime_error(
        "rendered prompt has fewer image placeholders than images were supplied");
  }
  return out;
}

}  // namespace r4dx::vision
