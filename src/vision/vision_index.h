// src/vision/vision_index.h -- the three index/weight tables `Qwen3_5VisionModel.forward` derives
// from `image_grid_thw` alone, before any weight is touched, plus the axial rope table they feed:
//
//   * the 4-tap bilinear gather that resamples the learned 48x48 position-embedding grid to this
//     image's patch grid (transformers.vision_utils.get_vision_interpolation_indices_and_weights
//     with mode="bilinear", align_corners=True, spatial_merge_size=2);
//   * the per-patch (h, w) position pair the vision rope rotates by
//     (transformers.vision_utils.get_vision_position_ids, the 2-axis variant -- NOT
//     Qwen3_5Model.get_vision_position_ids, which is the text-side 3-axis one in position_ids.h);
//   * cu_seqlens, the per-image attention segment boundaries r4d_attn_vit_h72_bf16 takes.
//
// All three walk patches in the same 2x2 spatial-merge-block-major order src/vision/preprocess.cpp
// emits them in, so an ordering mistake in any one of them silently mismatches the other two
// rather than failing loudly -- which is why each has its own assertion against a dumped reference
// tensor in tests/vision/test_vision_index.cpp rather than only being checked downstream.
#pragma once

#include <cstdint>
#include <vector>

#include "preprocess.h"

namespace r4dx::vision {

// 4 taps per patch (2 rows x 2 cols of the learned grid), flattened: patch p's taps are
// indices[p*4 + k] / weights[p*4 + k] for k in [0,4), with k = row_tap * 2 + col_tap. `indices`
// are flat offsets into the [num_grid_per_side^2, hidden] table.
struct PosEmbedInterpolation {
  std::vector<int32_t> indices;
  std::vector<float> weights;
  int64_t num_patches = 0;
};

PosEmbedInterpolation BuildPosEmbedInterpolation(const std::vector<GridThw>& grids,
                                                  int num_grid_per_side, int merge_size);

// [total_patches * 2] int32: patch p's (h_pos, w_pos) at [p*2] and [p*2+1].
std::vector<int32_t> BuildVisionRopePositionIds(const std::vector<GridThw>& grids, int merge_size);

// [num_segments + 1] int32, one segment per frame (t entries per image, so one per image for
// stills): the prefix sums of h*w. This is r4d_attn_vit_h72_bf16's `cu_seqlens` argument.
std::vector<int32_t> BuildCuSeqlens(const std::vector<GridThw>& grids);

// The axial rope table the encoder blocks are handed: cos/sin are each [total_patches, head_dim],
// laid out as cat([f(h), f(w), f(h), f(w)]) over head_dim/4 frequencies -- the FULL head rotates,
// unlike the text side's partial_rotary_factor=0.25. Computed in fp32 like the reference
// (Qwen3_5VisionRotaryEmbedding.forward upcasts before cos/sin).
void BuildVisionRopeCosSin(const std::vector<int32_t>& position_ids, int head_dim, double theta,
                            std::vector<float>* cos_out, std::vector<float>* sin_out);

}  // namespace r4dx::vision
