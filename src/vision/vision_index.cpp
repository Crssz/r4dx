#include "vision_index.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace r4dx::vision {
namespace {

// The reference decodes a patch's flat position back into (row, col) rather than iterating rows
// and columns, because the flat order is block-major, not raster:
//   in_col    = within % merge
//   in_row    = (within / merge) % merge
//   block_col = (within / merge^2) % blocks_w
//   block_row =  within / (merge^2 * blocks_w)
// This is the exact inverse of preprocess.cpp's PatchifyInto loop nest, and of
// get_vision_position_ids' reshape(h/m, m, w/m, m).transpose(1, 2).flatten().
void DecodeBlockMajor(int64_t within, int64_t merge, int64_t blocks_w, int64_t* row, int64_t* col) {
  const int64_t in_col = within % merge;
  const int64_t in_row = (within / merge) % merge;
  const int64_t block_col = (within / (merge * merge)) % blocks_w;
  const int64_t block_row = within / (merge * merge * blocks_w);
  *row = block_row * merge + in_row;
  *col = block_col * merge + in_col;
}

// One axis of `_interpolation_axis_taps_weights(mode="bilinear", align_corners=True,
// padding="border")`. align_corners=True means the target grid's endpoints map exactly onto the
// source table's endpoints -- the closed form of linspace(0, side-1, size)[index] -- and the
// clamp(size-1, min=1) guard is what keeps a single-patch axis from dividing by zero.
// Computed in float, not double: the reference casts the index to float32 first, and matching its
// rounding at the tap boundaries matters more here than extra precision would.
void AxisTapsWeights(int64_t index, int64_t size, int side, int32_t taps[2], float weights[2]) {
  const float src = static_cast<float>(index) * static_cast<float>(side - 1) /
                     static_cast<float>(std::max<int64_t>(size - 1, 1));
  const float floor_v = std::floor(src);
  const int32_t base = static_cast<int32_t>(floor_v);
  for (int k = 0; k < 2; ++k) {
    taps[k] = std::min(std::max(base + k, 0), side - 1);
    const float distance = std::fabs(src - floor_v - static_cast<float>(k));
    weights[k] = std::max(1.0f - distance, 0.0f);
  }
}

}  // namespace

PosEmbedInterpolation BuildPosEmbedInterpolation(const std::vector<GridThw>& grids,
                                                  int num_grid_per_side, int merge_size) {
  PosEmbedInterpolation out;
  int64_t total = 0;
  for (const GridThw& g : grids) total += g.PatchCount();
  out.num_patches = total;
  out.indices.resize(static_cast<size_t>(total) * 4);
  out.weights.resize(static_cast<size_t>(total) * 4);

  size_t cursor = 0;
  for (const GridThw& g : grids) {
    if (g.h % merge_size != 0 || g.w % merge_size != 0) {
      throw std::runtime_error("BuildPosEmbedInterpolation: grid is not a multiple of merge_size");
    }
    const int64_t frame_patches = g.h * g.w;
    const int64_t blocks_w = g.w / merge_size;
    for (int64_t i = 0; i < g.PatchCount(); ++i) {
      const int64_t within = i % frame_patches;  // h/w indices repeat across the t frames
      int64_t row = 0, col = 0;
      DecodeBlockMajor(within, merge_size, blocks_w, &row, &col);
      int32_t h_taps[2], w_taps[2];
      float h_w[2], w_w[2];
      AxisTapsWeights(row, g.h, num_grid_per_side, h_taps, h_w);
      AxisTapsWeights(col, g.w, num_grid_per_side, w_taps, w_w);
      for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
          out.indices[cursor + static_cast<size_t>(a) * 2 + b] =
              h_taps[a] * num_grid_per_side + w_taps[b];
          out.weights[cursor + static_cast<size_t>(a) * 2 + b] = h_w[a] * w_w[b];
        }
      }
      cursor += 4;
    }
  }
  return out;
}

std::vector<int32_t> BuildVisionRopePositionIds(const std::vector<GridThw>& grids, int merge_size) {
  int64_t total = 0;
  for (const GridThw& g : grids) total += g.PatchCount();
  std::vector<int32_t> out(static_cast<size_t>(total) * 2);

  size_t cursor = 0;
  for (const GridThw& g : grids) {
    const int64_t frame_patches = g.h * g.w;
    const int64_t blocks_w = g.w / merge_size;
    for (int64_t i = 0; i < g.PatchCount(); ++i) {
      int64_t row = 0, col = 0;
      DecodeBlockMajor(i % frame_patches, merge_size, blocks_w, &row, &col);
      out[cursor++] = static_cast<int32_t>(row);
      out[cursor++] = static_cast<int32_t>(col);
    }
  }
  return out;
}

std::vector<int32_t> BuildCuSeqlens(const std::vector<GridThw>& grids) {
  std::vector<int32_t> out;
  out.push_back(0);
  int64_t running = 0;
  for (const GridThw& g : grids) {
    for (int64_t frame = 0; frame < g.t; ++frame) {
      running += g.h * g.w;
      out.push_back(static_cast<int32_t>(running));
    }
  }
  return out;
}

void BuildVisionRopeCosSin(const std::vector<int32_t>& position_ids, int head_dim, double theta,
                            std::vector<float>* cos_out, std::vector<float>* sin_out) {
  if (head_dim % 4 != 0) throw std::runtime_error("BuildVisionRopeCosSin: head_dim must be a multiple of 4");
  if (position_ids.size() % 2 != 0) {
    throw std::runtime_error("BuildVisionRopeCosSin: position_ids must hold (h, w) pairs");
  }
  const int spatial_dim = head_dim / 2;   // h and w each rotate half the head
  const int num_freqs = spatial_dim / 2;  // head_dim/4 inverse frequencies
  const size_t tokens = position_ids.size() / 2;

  std::vector<float> inv_freq(static_cast<size_t>(num_freqs));
  for (int i = 0; i < num_freqs; ++i) {
    inv_freq[static_cast<size_t>(i)] = static_cast<float>(
        1.0 / std::pow(theta, static_cast<double>(2 * i) / static_cast<double>(spatial_dim)));
  }

  cos_out->resize(tokens * static_cast<size_t>(head_dim));
  sin_out->resize(tokens * static_cast<size_t>(head_dim));
  for (size_t p = 0; p < tokens; ++p) {
    float* cos_row = cos_out->data() + p * static_cast<size_t>(head_dim);
    float* sin_row = sin_out->data() + p * static_cast<size_t>(head_dim);
    for (int axis = 0; axis < 2; ++axis) {
      const float pos = static_cast<float>(position_ids[p * 2 + static_cast<size_t>(axis)]);
      for (int i = 0; i < num_freqs; ++i) {
        const float angle = pos * inv_freq[static_cast<size_t>(i)];
        const float c = std::cos(angle);
        const float s = std::sin(angle);
        const int slot = axis * num_freqs + i;
        // cat([f_h, f_w]) then cat([hw, hw]): the second half of the head repeats the first.
        cos_row[slot] = c;
        cos_row[slot + spatial_dim] = c;
        sin_row[slot] = s;
        sin_row[slot + spatial_dim] = s;
      }
    }
  }
}

}  // namespace r4dx::vision
