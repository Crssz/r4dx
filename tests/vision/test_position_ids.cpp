// tests/vision/test_position_ids.cpp -- r4dx::vision::BuildMropePositionIds against the REAL
// Qwen3_5Model.get_rope_index (tools/reference/golden_out/rope_index.safetensors, produced by
// tools/reference/rope_index_golden.py, which calls the unmodified reference method).
//
// Every one of the golden's five cases is compared element by element, not statistically: text
// only, text + 1 image, text + 2 images with different grids, an image at sequence position 0, and
// an image as the final tokens. The decode-side rope delta is checked too, since that is the value
// that makes the FIRST token after an image prompt land in the right place and is invisible to any
// check that only looks at the prompt's own position rows.
#include <cstdio>
#include <string>
#include <vector>

#include "position_ids.h"
#include "preprocess.h"
#include "vision_test_common.h"

using namespace r4dx_vision_test;
using r4dx::vision::BuildMropePositionIds;
using r4dx::vision::GridThw;
using r4dx::vision::kMmTokenTypeImage;
using r4dx::vision::MmTokenTypeIdsFromTokens;

namespace {

const char* kGoldenPath = R4DX_SOURCE_DIR "/tools/reference/golden_out/rope_index.safetensors";

// docs/vision.md "Model facts": config.json's top-level image/video token ids. Derived from the
// golden below as well, so a checkpoint change would fail here rather than silently disagree.
constexpr int32_t kImageTokenId = 248056;
constexpr int32_t kVideoTokenId = 248057;
constexpr int kMergeSize = 2;

void CheckCase(const r4dx_convert::SafetensorsReader& golden, const std::string& name) {
  const auto input_ids64 = ReadInt(golden, name + ".input_ids");
  const auto types64 = ReadInt(golden, name + ".mm_token_type_ids");
  const auto ref_positions = ReadInt(golden, name + ".position_ids");
  const auto ref_delta = ReadInt(golden, name + ".mrope_position_deltas");
  const int64_t seq_len = static_cast<int64_t>(input_ids64.size());

  std::vector<int32_t> input_ids(input_ids64.begin(), input_ids64.end());
  std::vector<uint8_t> types(types64.begin(), types64.end());

  // Every image-typed position must carry the documented image token id; this is the golden
  // telling us the id rather than the test asserting one it made up.
  for (size_t i = 0; i < types.size(); ++i) {
    if (types[i] == kMmTokenTypeImage) CHECK(input_ids[i] == kImageTokenId);
  }
  // The production path derives the type ids from the token ids; it must reproduce the reference
  // processor's own mm_token_type_ids exactly.
  CHECK(MmTokenTypeIdsFromTokens(input_ids, kImageTokenId, kVideoTokenId) == types);

  std::vector<GridThw> grids;
  if (golden.Has(name + ".image_grid_thw")) {
    const auto g = ReadInt(golden, name + ".image_grid_thw");
    for (size_t i = 0; i + 2 < g.size(); i += 3) grids.push_back(GridThw{g[i], g[i + 1], g[i + 2]});
  }

  const auto got = BuildMropePositionIds(types, grids, kMergeSize);
  CHECK(got.seq_len == seq_len);
  CHECK(static_cast<int64_t>(ref_positions.size()) == 3 * seq_len);

  int64_t mismatches = 0, first_bad = -1;
  for (int64_t axis = 0; axis < 3 && static_cast<int64_t>(ref_positions.size()) == 3 * seq_len; ++axis) {
    for (int64_t p = 0; p < seq_len; ++p) {
      // The reference's shape is [3, batch=1, seq], so its flat layout matches [3, seq] here.
      const int64_t ref = ref_positions[static_cast<size_t>(axis * seq_len + p)];
      if (got.At(static_cast<int>(axis), p) != ref) {
        ++mismatches;
        if (first_bad < 0) {
          first_bad = p;
          std::fprintf(stderr, "  %s: axis %lld token %lld: got %d golden %lld\n", name.c_str(),
                       static_cast<long long>(axis), static_cast<long long>(p),
                       got.At(static_cast<int>(axis), p), static_cast<long long>(ref));
        }
      }
    }
  }
  CHECK(mismatches == 0);
  CHECK(!ref_delta.empty() && got.mrope_position_delta == ref_delta[0]);

  std::printf("[test_position_ids] %-34s seq_len=%lld images=%zu delta=%lld mismatches=%lld\n",
              name.c_str(), static_cast<long long>(seq_len), grids.size(),
              static_cast<long long>(got.mrope_position_delta), static_cast<long long>(mismatches));
}

// The CONTINUATION property the multi-turn/prefix-reuse path depends on (docs/vision.md): feeding
// a prompt in two pieces -- the second built with (seq_start, mrope_start) carried from the
// first's own delta, exactly as Model::PrefillMultimodal does for turn 2 of a chat -- must produce
// the identical position rows and the identical final delta as building the whole thing at once.
// Split points are chosen at text tokens only, because an image run is atomic by construction (a
// span carries its whole merged grid); a split inside one is a caller bug, not a case to support.
void CheckContinuationSplits(const r4dx_convert::SafetensorsReader& golden,
                              const std::string& name) {
  const auto types64 = ReadInt(golden, name + ".mm_token_type_ids");
  std::vector<uint8_t> types(types64.begin(), types64.end());
  const int64_t seq_len = static_cast<int64_t>(types.size());

  std::vector<GridThw> grids;
  if (golden.Has(name + ".image_grid_thw")) {
    const auto g = ReadInt(golden, name + ".image_grid_thw");
    for (size_t i = 0; i + 2 < g.size(); i += 3) grids.push_back(GridThw{g[i], g[i + 1], g[i + 2]});
  }
  const auto whole = BuildMropePositionIds(types, grids, kMergeSize);

  int64_t splits_checked = 0, mismatches = 0;
  for (int64_t split = 1; split < seq_len; ++split) {
    // Only split between runs: `split` must be a text token AND the token before it must be text
    // (otherwise the split sits immediately after an image, which IS legal, so allow that too --
    // what is not allowed is splitting a placeholder run in half).
    if (types[static_cast<size_t>(split)] == kMmTokenTypeImage &&
        types[static_cast<size_t>(split - 1)] == kMmTokenTypeImage) {
      continue;
    }
    const std::vector<uint8_t> head(types.begin(), types.begin() + static_cast<ptrdiff_t>(split));
    const std::vector<uint8_t> tail(types.begin() + static_cast<ptrdiff_t>(split), types.end());
    // Split the grid list at whichever images fall entirely in the head.
    size_t head_images = 0;
    for (int64_t p = 0; p < split; ++p) {
      const bool run_start = types[static_cast<size_t>(p)] == kMmTokenTypeImage &&
                              (p == 0 || types[static_cast<size_t>(p - 1)] != kMmTokenTypeImage);
      if (run_start) ++head_images;
    }
    const std::vector<GridThw> head_grids(grids.begin(),
                                           grids.begin() + static_cast<ptrdiff_t>(head_images));
    const std::vector<GridThw> tail_grids(grids.begin() + static_cast<ptrdiff_t>(head_images),
                                           grids.end());

    const auto a = BuildMropePositionIds(head, head_grids, kMergeSize);
    const auto b = BuildMropePositionIds(tail, tail_grids, kMergeSize, /*seq_start=*/split,
                                          /*mrope_start=*/split + a.mrope_position_delta);
    ++splits_checked;
    for (int axis = 0; axis < 3; ++axis) {
      for (int64_t p = 0; p < split; ++p) {
        if (a.At(axis, p) != whole.At(axis, p)) ++mismatches;
      }
      for (int64_t p = 0; p < seq_len - split; ++p) {
        if (b.At(axis, p) != whole.At(axis, split + p)) ++mismatches;
      }
    }
    if (b.mrope_position_delta != whole.mrope_position_delta) ++mismatches;
  }
  CHECK(mismatches == 0);
  std::printf("[test_position_ids] %-34s %lld continuation split(s), %lld mismatch(es)\n",
              name.c_str(), static_cast<long long>(splits_checked),
              static_cast<long long>(mismatches));
}

// An image run whose length does not match its grid's merged token count, and a video token, are
// both caller bugs that must surface as exceptions rather than as silently shifted positions.
void CheckRejectsInconsistentInput() {
  std::vector<uint8_t> types(10, 0);
  for (int i = 3; i < 7; ++i) types[static_cast<size_t>(i)] = kMmTokenTypeImage;
  bool threw = false;
  try {
    BuildMropePositionIds(types, {GridThw{1, 28, 28}}, kMergeSize);  // 196 merged != 4 tokens
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);

  threw = false;
  try {
    std::vector<uint8_t> video(4, r4dx::vision::kMmTokenTypeVideo);
    BuildMropePositionIds(video, {}, kMergeSize);
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);
}

}  // namespace

int main() {
  if (!FileExists(kGoldenPath)) return SkipMissing(kGoldenPath);
  try {
    const r4dx_convert::SafetensorsReader golden(r4dx_convert::Utf8ToWide(kGoldenPath));
    for (const char* name : {"text_only", "text_one_image", "text_two_images_different_grids",
                              "image_first", "image_last"}) {
      CheckCase(golden, name);
      CheckContinuationSplits(golden, name);
    }
    CheckRejectsInconsistentInput();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "test_position_ids: unexpected exception: %s\n", e.what());
    return 1;
  }
  if (g_failures != 0) {
    std::fprintf(stderr, "test_position_ids: %d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("test_position_ids: OK\n");
  return 0;
}
