#pragma once

#include <cstdint>
#include <vector>

#include "image_decode.h"
#include "preprocess.h"

namespace r4dx::vision {

constexpr int kInkling2NativePatch = 20;
constexpr int kInkling2MergeSize = 2;
constexpr int kInkling2Factor = kInkling2NativePatch * kInkling2MergeSize;  // 40
constexpr int kInkling2PatchDim = kInkling2NativePatch * kInkling2NativePatch * 3;  // 1200
constexpr int kInkling2ImageTokenId = 200054;  // <|unused_200054|>

struct Inkling2ImageConfig {
  int64_t max_pixels = 1024LL * 1024LL;  // default cap (<= 640 merged tokens; up to 4096*4096)
  int64_t max_side = 1280;               // default max side (32x32 merged = 1024 tokens)
};

struct Inkling2Preprocessed {
  int64_t grid_h = 0, grid_w = 0;       // native 20px patch grid (both even)
  int64_t resized_h = 0, resized_w = 0;
  std::vector<float> pixel_values;      // [grid_h * grid_w, 1200] HWC float32 in [-1, 1]
};

Inkling2Preprocessed PreprocessInkling2Image(const DecodedImage& img,
                                             const Inkling2ImageConfig& cfg = Inkling2ImageConfig());

}  // namespace r4dx::vision
