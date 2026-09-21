// src/vision/position_ids.h -- the TEXT-side mrope position ids for a prompt containing images,
// i.e. `Qwen3_5Model.get_rope_index` (docs/vision.md "Text-side splicing"). Host-side bookkeeping,
// not a kernel: the existing rope_partial_mrope kernel (docs/architecture.md) already consumes the
// [11,11,10]-section (t,h,w) rows this produces.
//
// The rule that is easy to get wrong, and whose failure mode is plausible-looking garbage rather
// than a crash, is the advance after an image: `current_pos += max(grid_h, grid_w) / merge_size`
// -- the larger merged SPATIAL extent, not the image's token count. A 28x28-patch image occupies
// 196 sequence positions but only advances mrope position space by 14. Getting that wrong shifts
// every position after the first image.
//
// Video (mm_token_type_id 2) is rejected rather than half-implemented: r4dx has no video path, and
// the reference's video handling (splitting a t>1 entry into t separate grids before the walk) has
// no golden here to test against.
#pragma once

#include <cstdint>
#include <vector>

#include "preprocess.h"

namespace r4dx::vision {

constexpr uint8_t kMmTokenTypeText = 0;
constexpr uint8_t kMmTokenTypeImage = 1;
constexpr uint8_t kMmTokenTypeVideo = 2;

struct MropePositions {
  // [3 * seq_len] int32, row-major by axis: the t row occupies [0, seq_len), h [seq_len,
  // 2*seq_len), w [2*seq_len, 3*seq_len) -- the same [3, seq] layout the reference returns (with
  // its batch dimension of 1 dropped, since r4dx builds one sequence at a time).
  std::vector<int32_t> position_ids;
  int64_t seq_len = 0;
  // position_ids.max() + 1 - seq_len. Decode step i (0-based, immediately after this prompt) uses
  // position `seq_len + i + mrope_position_delta` on all three rows.
  //
  // For a CONTINUATION build (non-zero `seq_start`, see BuildMropePositionIds' overload) this is
  // still the delta a token at absolute sequence index `s` adds to get its mrope position, i.e.
  // `next_mrope_pos - (seq_start + seq_len)` -- the same quantity, expressed so that it stays
  // valid for every later token of the conversation rather than only for this block.
  int64_t mrope_position_delta = 0;

  int32_t At(int axis, int64_t pos) const {
    return position_ids[static_cast<size_t>(axis) * static_cast<size_t>(seq_len) +
                        static_cast<size_t>(pos)];
  }
};

// `mm_token_type_ids` is one entry per prompt token: kMmTokenTypeText for everything including the
// <vision_start>/<vision_end> markers, kMmTokenTypeImage only for the image-placeholder tokens.
// `image_grids` must carry one entry per image run, in order, each with
// MergedTokenCount(merge_size) equal to that run's length.
//
// `seq_start`/`mrope_start` (vision milestone stage 4) extend the reference's own walk to a
// CONTINUATION block -- the second and later turns of a chat, or any prefill that is not the start
// of the conversation. `seq_start` is the absolute sequence index of `mm_token_type_ids[0]` and
// `mrope_start` is the mrope position the walk resumes at (== `seq_start + previous delta`).
// (0, 0) is the whole-prompt case and reproduces the reference exactly; the returned
// `position_ids` always covers only this block, indexed from 0.
MropePositions BuildMropePositionIds(const std::vector<uint8_t>& mm_token_type_ids,
                                      const std::vector<GridThw>& image_grids, int merge_size,
                                      int64_t seq_start = 0, int64_t mrope_start = 0);

// Convenience for callers holding a token id sequence: marks every `image_token_id` as an image
// token. Throws if a `video_token_id` is present, matching BuildMropePositionIds' own refusal.
std::vector<uint8_t> MmTokenTypeIdsFromTokens(const std::vector<int32_t>& input_ids,
                                               int32_t image_token_id, int32_t video_token_id);

}  // namespace r4dx::vision
