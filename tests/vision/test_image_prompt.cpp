// tests/vision/test_image_prompt.cpp -- r4dx::vision::ExpandImagePlaceholders (image_prompt.h),
// the routine that decides each image's splice offset and run length for BOTH user-facing entry
// points (r4dx-cli's --image, the server's image_url content parts) and for
// tests/vision/tool_vision_chat.
//
// It needs no golden, no container and no GPU: the contract is pure integer bookkeeping over a
// token vector, so this test builds the inputs by hand and asserts the exact expanded sequence and
// the exact spans. That is worth a test of its own because the failure it guards against is
// silent -- an off-by-one in `offset` or `tokens` writes merged image rows over real text
// embeddings, and Model::PrefillMultimodal's own placeholder-id check cannot catch a span that is
// merely mis-SIZED in the same direction as the placeholder run it sits in.
#include <cstdio>
#include <string>
#include <vector>

#include "image_prompt.h"
#include "preprocess.h"
#include "vision_test_common.h"

using r4dx::vision::ExpandedImagePrompt;
using r4dx::vision::ExpandImagePlaceholders;
using r4dx::vision::GridThw;
using r4dx::vision::ImagePlaceholderSpan;
using r4dx_vision_test::g_failures;

namespace {

// Qwen3.8-27B's own `<|image_pad|>` id and merge_size (docs/vision.md "Model facts"). Nothing here
// depends on the specific value -- it just has to be the id the caller passes -- but using the
// real one keeps the expanded sequences below readable against a real rendered prompt.
constexpr int32_t kPad = 248056;
constexpr int kMergeSize = 2;

// Stand-ins for a rendered prompt's surrounding text tokens. Deliberately NOT adjacent to kPad, so
// an expansion that ran one token long or short shows up as a changed text token, not just a
// changed count.
constexpr int32_t kVisionStart = 100;
constexpr int32_t kVisionEnd = 101;

ImagePlaceholderSpan MakeImage(int64_t h, int64_t w, const uint16_t* embeds) {
  ImagePlaceholderSpan sp;
  sp.grid = GridThw{1, h, w};
  sp.embeds = embeds;
  return sp;
}

// Two distinct non-null device-pointer stand-ins: ExpandImagePlaceholders never dereferences
// `embeds`, it only has to carry each image's OWN pointer through to that image's OWN span, which
// is exactly what a two-image request would get wrong by reusing the first image's rows.
const uint16_t* const kEmbedsA = reinterpret_cast<const uint16_t*>(0x1000);
const uint16_t* const kEmbedsB = reinterpret_cast<const uint16_t*>(0x2000);

void CheckTokens(const std::vector<int32_t>& got, const std::vector<int32_t>& want,
                 const char* what) {
  CHECK(got.size() == want.size());
  if (got.size() != want.size()) {
    std::fprintf(stderr, "  %s: got %zu tokens, want %zu\n", what, got.size(), want.size());
    return;
  }
  for (size_t i = 0; i < want.size(); ++i) {
    if (got[i] != want[i]) {
      std::fprintf(stderr, "  %s: token %zu = %d, want %d\n", what, i, got[i], want[i]);
      CHECK(got[i] == want[i]);
      return;
    }
  }
}

// One image, placeholder at the END of the prompt (the common `--prompt "Describe this."` shape
// once the template has appended nothing after the vision block).
void TestSingleImage() {
  // grid 4x6 -> (4/2)*(6/2) = 6 merged tokens.
  const std::vector<int32_t> raw = {1, 2, kVisionStart, kPad, kVisionEnd};
  const std::vector<ImagePlaceholderSpan> images = {MakeImage(4, 6, kEmbedsA)};
  const ExpandedImagePrompt out = ExpandImagePlaceholders(raw, kPad, images, kMergeSize);

  CheckTokens(out.tokens,
              {1, 2, kVisionStart, kPad, kPad, kPad, kPad, kPad, kPad, kVisionEnd}, "single");
  CHECK(out.spans.size() == 1);
  if (out.spans.size() != 1) return;
  CHECK(out.spans[0].offset == 3);
  CHECK(out.spans[0].tokens == 6);
  CHECK(out.spans[0].grid.h == 4 && out.spans[0].grid.w == 6);
  CHECK(out.spans[0].embeds == kEmbedsA);
}

// An image in the MIDDLE, with real text tokens on both sides: the case where a mis-sized run
// silently overwrites following text rather than running off the end of the sequence.
void TestImageInMiddle() {
  // grid 2x2 -> 1 merged token: the smallest run there is, so an off-by-one is a 100% error.
  const std::vector<int32_t> raw = {7, kPad, 8, 9};
  const std::vector<ImagePlaceholderSpan> images = {MakeImage(2, 2, kEmbedsA)};
  const ExpandedImagePrompt out = ExpandImagePlaceholders(raw, kPad, images, kMergeSize);

  CheckTokens(out.tokens, {7, kPad, 8, 9}, "middle-1tok");
  CHECK(out.spans.size() == 1);
  if (out.spans.size() != 1) return;
  CHECK(out.spans[0].offset == 1);
  CHECK(out.spans[0].tokens == 1);

  // Same shape, a wider image: the tail text has to shift by exactly (tokens - 1).
  const std::vector<ImagePlaceholderSpan> wide = {MakeImage(2, 8, kEmbedsA)};  // 1*4 = 4 tokens
  const ExpandedImagePrompt out2 = ExpandImagePlaceholders(raw, kPad, wide, kMergeSize);
  CheckTokens(out2.tokens, {7, kPad, kPad, kPad, kPad, 8, 9}, "middle-4tok");
  CHECK(out2.spans.size() == 1);
  if (out2.spans.size() != 1) return;
  CHECK(out2.spans[0].offset == 1);
  CHECK(out2.spans[0].tokens == 4);
}

// Two images of DIFFERENT sizes, text between them: the second span's offset must be measured in
// the EXPANDED sequence (i.e. shifted by the first image's whole run), which is the bug a
// raw-token offset would produce.
void TestTwoImages() {
  const std::vector<int32_t> raw = {1, kPad, 2, 3, kPad, 4};
  const std::vector<ImagePlaceholderSpan> images = {
      MakeImage(2, 4, kEmbedsA),   // 1*2 = 2 tokens
      MakeImage(4, 6, kEmbedsB),   // 2*3 = 6 tokens
  };
  const ExpandedImagePrompt out = ExpandImagePlaceholders(raw, kPad, images, kMergeSize);

  CheckTokens(out.tokens,
              {1, kPad, kPad, 2, 3, kPad, kPad, kPad, kPad, kPad, kPad, 4}, "two-images");
  CHECK(out.spans.size() == 2);
  if (out.spans.size() != 2) return;
  CHECK(out.spans[0].offset == 1);
  CHECK(out.spans[0].tokens == 2);
  CHECK(out.spans[0].embeds == kEmbedsA);
  // 1 (raw prefix) + 2 (image 0's run) + 2 (the text between) == 5.
  CHECK(out.spans[1].offset == 5);
  CHECK(out.spans[1].tokens == 6);
  CHECK(out.spans[1].embeds == kEmbedsB);
  // Every span must cover only placeholder ids -- the invariant PrefillMultimodal re-checks.
  for (const auto& sp : out.spans) {
    for (int64_t i = sp.offset; i < sp.offset + sp.tokens; ++i) {
      CHECK(out.tokens[static_cast<size_t>(i)] == kPad);
    }
  }
}

// A prompt with no image at all must come back byte-identical and span-free: the text-only path
// through the server and the CLI both call this routine unconditionally.
void TestNoImages() {
  const std::vector<int32_t> raw = {1, 2, 3, 4, 5};
  const ExpandedImagePrompt out = ExpandImagePlaceholders(raw, kPad, {}, kMergeSize);
  CheckTokens(out.tokens, raw, "no-images");
  CHECK(out.spans.empty());
}

bool Throws(const std::vector<int32_t>& raw, const std::vector<ImagePlaceholderSpan>& images) {
  try {
    ExpandImagePlaceholders(raw, kPad, images, kMergeSize);
  } catch (const std::runtime_error&) {
    return true;
  }
  return false;
}

// Both mismatch directions must throw rather than splice the wrong image's rows (or none).
void TestMismatchThrows() {
  const std::vector<ImagePlaceholderSpan> one = {MakeImage(2, 2, kEmbedsA)};
  const std::vector<ImagePlaceholderSpan> two = {MakeImage(2, 2, kEmbedsA),
                                                  MakeImage(2, 2, kEmbedsB)};
  CHECK(Throws({1, kPad, kPad, 2}, one));   // more placeholders than images
  CHECK(Throws({1, kPad, 2}, two));         // fewer placeholders than images
  CHECK(Throws({1, 2, 3}, one));            // no placeholder at all, one image supplied
  CHECK(!Throws({1, kPad, 2}, one));        // the matching case still succeeds
}

}  // namespace

int main() {
  TestSingleImage();
  TestImageInMiddle();
  TestTwoImages();
  TestNoImages();
  TestMismatchThrows();

  if (g_failures != 0) {
    std::fprintf(stderr, "test_image_prompt: %d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("test_image_prompt: OK\n");
  return 0;
}
