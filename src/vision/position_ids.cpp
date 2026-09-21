#include "position_ids.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace r4dx::vision {

MropePositions BuildMropePositionIds(const std::vector<uint8_t>& mm_token_type_ids,
                                      const std::vector<GridThw>& image_grids, int merge_size,
                                      int64_t seq_start, int64_t mrope_start) {
  if (merge_size <= 0) throw std::runtime_error("BuildMropePositionIds: merge_size must be positive");
  if (seq_start < 0 || mrope_start < 0) {
    throw std::runtime_error("BuildMropePositionIds: seq_start/mrope_start must be non-negative");
  }
  MropePositions out;
  out.seq_len = static_cast<int64_t>(mm_token_type_ids.size());
  out.position_ids.assign(static_cast<size_t>(out.seq_len) * 3, 0);
  if (out.seq_len == 0) {
    out.mrope_position_delta = mrope_start - seq_start;
    return out;
  }

  int32_t* row_t = out.position_ids.data();
  int32_t* row_h = row_t + out.seq_len;
  int32_t* row_w = row_h + out.seq_len;

  int64_t current_pos = mrope_start;
  size_t next_image = 0;
  int64_t i = 0;
  while (i < out.seq_len) {
    const uint8_t type = mm_token_type_ids[static_cast<size_t>(i)];
    int64_t run_end = i;
    while (run_end < out.seq_len && mm_token_type_ids[static_cast<size_t>(run_end)] == type) ++run_end;
    const int64_t run_len = run_end - i;

    if (type == kMmTokenTypeText) {
      for (int64_t k = 0; k < run_len; ++k) {
        const int32_t p = static_cast<int32_t>(current_pos + k);
        row_t[i + k] = p;
        row_h[i + k] = p;
        row_w[i + k] = p;
      }
      current_pos += run_len;
    } else if (type == kMmTokenTypeImage) {
      if (next_image >= image_grids.size()) {
        throw std::runtime_error("BuildMropePositionIds: more image runs than image_grids entries");
      }
      const GridThw& grid = image_grids[next_image++];
      const int64_t llm_h = grid.h / merge_size;
      const int64_t llm_w = grid.w / merge_size;
      const int64_t llm_t = grid.t;  // temp_merge_size is 1 for this model
      if (llm_t * llm_h * llm_w != run_len) {
        throw std::runtime_error(
            "BuildMropePositionIds: image run of " + std::to_string(run_len) +
            " tokens does not match its grid's merged token count " +
            std::to_string(llm_t * llm_h * llm_w));
      }
      // Qwen3_5Model.get_vision_position_ids(current_pos, grid, 1, merge_size): a (t,h,w) meshgrid
      // over the MERGED grid, every axis offset by current_pos (the t axis additionally carries
      // its own arange, which is a constant 0 for a still image).
      int64_t k = 0;
      for (int64_t t = 0; t < llm_t; ++t) {
        for (int64_t h = 0; h < llm_h; ++h) {
          for (int64_t w = 0; w < llm_w; ++w, ++k) {
            row_t[i + k] = static_cast<int32_t>(t + current_pos);
            row_h[i + k] = static_cast<int32_t>(h + current_pos);
            row_w[i + k] = static_cast<int32_t>(w + current_pos);
          }
        }
      }
      // The advance rule this whole file exists for -- see the header comment. Note that it also
      // makes `current_pos` the reference's own `position_ids.max() + 1`: the image's largest
      // emitted position is `current_pos + max(llm_t, llm_h, llm_w) - 1`, and
      // `max(grid.h, grid.w) / merge_size == max(llm_h, llm_w) >= llm_t (== 1 here)`, so the two
      // agree exactly -- which is what lets the delta below be a running-counter difference
      // rather than a separate max scan.
      current_pos += std::max(grid.h, grid.w) / merge_size;
    } else {
      throw std::runtime_error("BuildMropePositionIds: video tokens (mm_token_type_id 2) are not "
                                "supported; r4dx has no video path");
    }
    i = run_end;
  }

  if (next_image != image_grids.size()) {
    throw std::runtime_error("BuildMropePositionIds: " + std::to_string(image_grids.size()) +
                              " image grids supplied but only " + std::to_string(next_image) +
                              " image runs found in mm_token_type_ids");
  }
  out.mrope_position_delta = current_pos - (seq_start + out.seq_len);
  return out;
}

std::vector<uint8_t> MmTokenTypeIdsFromTokens(const std::vector<int32_t>& input_ids,
                                               int32_t image_token_id, int32_t video_token_id) {
  std::vector<uint8_t> out(input_ids.size(), kMmTokenTypeText);
  for (size_t i = 0; i < input_ids.size(); ++i) {
    if (input_ids[i] == image_token_id) {
      out[i] = kMmTokenTypeImage;
    } else if (input_ids[i] == video_token_id) {
      throw std::runtime_error("MmTokenTypeIdsFromTokens: video token found; r4dx has no video path");
    }
  }
  return out;
}

}  // namespace r4dx::vision
