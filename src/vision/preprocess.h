// src/vision/preprocess.h -- host-side image preprocessing: decoded 8-bit RGB -> the exact
// `pixel_values` / `image_grid_thw` pair `Qwen2VLImageProcessor` produces, which is what the
// vision tower's patch-embed GEMM consumes (docs/vision.md).
//
// "Exact" is meant literally. The reference pipeline is integer-deterministic end to end -- PIL
// RGB conversion, smart_resize, an integer uint8 resampler, then one fp32 affine per pixel -- so
// the C++ side is not approximating it: tests/vision/test_preprocess.cpp asserts bit equality of
// the patch tensor against tools/reference/golden_out/vision_tower*.safetensors, not a tolerance.
// The two places that are easy to get subtly wrong, and that a "looks like an image" eyeball test
// would never catch, are:
//
//   1. The resampler. It is NOT a float bilinear resize. transformers' Qwen2VLImageProcessor sets
//      resample=PILImageResampling.BICUBIC and runs it through TorchvisionBackend.resize, i.e.
//      tvF.resize(uint8_tensor, BICUBIC, antialias=True) on the *uint8* tensor, which dispatches
//      to torch's uint8 antialias kernel: two separable passes (horizontal first) with an
//      intermediate buffer that is itself rounded and clamped back to uint8, int16 fixed-point
//      weights, and Pillow's cubic kernel with a = -0.5 -- not the a = -0.75 of torch's
//      non-antialiased bicubic. Getting a = -0.75, or keeping a float intermediate, or running the
//      vertical pass first, each produces an image that looks identical and is wrong by up to ~24
//      of 255 per pixel on real content (measured -- see docs/vision.md "Resampling"). ResizeU8 below
//      reproduces the real algorithm; its own comments carry the per-step derivation.
//   2. The patch order. `pixel_values` rows are NOT in raster order: they are 2x2
//      spatial-merge-block-major, which is what lets the tower's merger do a plain view() with no
//      permute. PatchifyInto() below is the only place that ordering is expressed.
#pragma once

#include <cstdint>
#include <vector>

#include "image_decode.h"

namespace r4dx::vision {

// One image's patch grid. `t` is the temporal extent (always 1 for a still image; the frame is
// duplicated temporal_patch_size times *inside* each patch row rather than becoming a second grid
// step), `h`/`w` are counts of patch_size-sized patches, always even because smart_resize rounds
// each side to a multiple of patch_size * merge_size.
struct GridThw {
  int64_t t = 1;
  int64_t h = 0;
  int64_t w = 0;

  int64_t PatchCount() const { return t * h * w; }
  int64_t MergedTokenCount(int64_t merge_size) const {
    return t * (h / merge_size) * (w / merge_size);
  }
};

// Defaults are this checkpoint's real preprocessor_config.json values (C:\AI\models\Qwen3.8-27B):
// note that shortest_edge/longest_edge are TOTAL PIXEL COUNTS (h*w bounds), not side lengths.
struct ImageProcessorConfig {
  int patch_size = 16;
  int temporal_patch_size = 2;
  int merge_size = 2;
  int64_t min_pixels = 65536;       // size.shortest_edge
  int64_t max_pixels = 16777216;    // size.longest_edge
  float image_mean[3] = {0.5f, 0.5f, 0.5f};
  float image_std[3] = {0.5f, 0.5f, 0.5f};
  // double, not float: the reference's rescale_factor is a Python float, and the fused constants
  // below are `image_mean * (1.0 / rescale_factor)` evaluated with that double reciprocal before
  // it is narrowed. Storing 1/255 as a float instead lands the fused mean on 127.499992 rather
  // than exactly 127.5 -- one fp32 ulp off on ~95% of pixels.
  double rescale_factor = 1.0 / 255.0;

  int Factor() const { return patch_size * merge_size; }
  int64_t PatchDim() const {
    return 3LL * temporal_patch_size * patch_size * patch_size;  // 1536 for this checkpoint
  }

  // TorchvisionBackend::_fuse_mean_std_and_rescale_factor: the per-channel affine actually applied
  // to the raw byte, as (byte - mean) / std in fp32. 127.5 / 127.5 for this checkpoint.
  void FusedMeanStd(float mean[3], float std[3]) const {
    const float inv_rescale = static_cast<float>(1.0 / rescale_factor);
    for (int c = 0; c < 3; ++c) {
      mean[c] = image_mean[c] * inv_rescale;
      std[c] = image_std[c] * inv_rescale;
    }
  }
};

struct PreprocessedImages {
  // [total_patches, patch_dim], row-major; total_patches is the sum over images of
  // grid.PatchCount(), so a multi-image request produces one concatenated tensor exactly as the
  // reference processor does.
  std::vector<float> pixel_values;
  int64_t patch_dim = 0;
  std::vector<GridThw> grid_thw;  // one row per image, in input order

  int64_t TotalPatches() const {
    return patch_dim == 0 ? 0 : static_cast<int64_t>(pixel_values.size()) / patch_dim;
  }
};

// transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize, including its Python
// semantics: banker's rounding (round-half-to-even, which is what Python's round() does and what a
// naive lround() would get wrong on exact .5 ratios) and the >200 aspect-ratio rejection.
// Throws std::runtime_error on that rejection, like the reference raises ValueError.
void SmartResize(int64_t height, int64_t width, int64_t factor, int64_t min_pixels,
                  int64_t max_pixels, int64_t* out_height, int64_t* out_width);

// tvF.resize(uint8, BICUBIC, antialias=True) -- see the file comment. Returns `src` unchanged when
// the size already matches, which is also what torchvision does (and matters: the 448x448 golden
// case never resizes at all).
DecodedImage ResizeU8(const DecodedImage& src, int out_width, int out_height);

// Decode -> smart_resize -> resample -> rescale+normalize -> patchify, for one or more images.
PreprocessedImages PreprocessImages(const std::vector<DecodedImage>& images,
                                     const ImageProcessorConfig& cfg);

// This checkpoint's processor config, with `image_max_pixels` substituted for its own `max_pixels`
// ceiling (`--image-max-pixels` on both binaries, docs/vision.md "Large images"). 0 keeps the
// checkpoint's own 16777216-pixel ceiling.
//
// Note what a cap does and does NOT do: `smart_resize` already contains the "too many pixels"
// branch, and it DOWNSIZES (scales both sides by sqrt(h*w / max_pixels), then rounds each down to
// a multiple of patch_size * merge_size) rather than rejecting. Lowering the ceiling therefore
// costs a large attachment resolution, never an error -- which is the whole point of putting the
// cap here instead of in a request validator. Throws on a negative value.
ImageProcessorConfig MakeImageProcessorConfig(int64_t image_max_pixels);

}  // namespace r4dx::vision
