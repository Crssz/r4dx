#include "preprocess.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace r4dx::vision {
namespace {

// Python's round() is round-half-to-even, and smart_resize's `round(height / factor)` hits an
// exact .5 whenever a side is an odd multiple of factor/2 (e.g. 48 with factor 32) -- common
// enough in real images that a plain floor(x + 0.5) would silently pick a different grid than the
// reference for those sizes.
int64_t RoundHalfToEven(double x) {
  const double r = std::nearbyint(x);  // nearbyint honours the default FE_TONEAREST = half-to-even
  return static_cast<int64_t>(r);
}

// ---------------------------------------------------------------------------------------------
// The resampler. Everything from here to ResizeU8 is a transcription of torch's uint8 antialias
// path (aten's UpSampleKernel.cpp `HelperInterpCubic<uint8_t, int16_t>` +
// `_separable_upsample_generic_Nd_kernel_impl`), which is what tvF.resize(uint8, BICUBIC,
// antialias=True) dispatches to on CPU. Verified against the reference on random-noise images (the
// worst case for any rounding disagreement): zero differing bytes out of 2.8 M.
// ---------------------------------------------------------------------------------------------

// Pillow's cubic convolution kernel. a = -0.5, NOT the a = -0.75 used by torch's *non*-antialiased
// bicubic -- the antialias path deliberately follows Pillow (torch's own comment cites
// Resample.c). Reproducing the -0.75 variant here is a ~4% weight error that shows up as up to 24
// levels of 255 on real content.
double CubicFilter(double x) {
  constexpr double a = -0.5;
  x = std::fabs(x);
  if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
  if (x < 2.0) return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
  return 0.0;
}

// Per-output-index gather window and int16 weights for one axis.
struct AxisWeights {
  int out_size = 0;
  int max_taps = 0;
  int precision = 0;           // weights are round(w * (1 << precision))
  std::vector<int> begin;      // [out_size] first source index of the window
  std::vector<int> taps;       // [out_size] window length
  std::vector<int32_t> w;      // [out_size * max_taps]
};

AxisWeights BuildAxisWeights(int in_size, int out_size) {
  AxisWeights a;
  a.out_size = out_size;
  const double scale = static_cast<double>(in_size) / static_cast<double>(out_size);
  // interp_size is 4 for cubic; antialiasing widens the kernel's support by the downscale factor
  // (and only when downscaling), which is what actually low-passes the image.
  const double support = (scale >= 1.0) ? 2.0 * scale : 2.0;
  const double invscale = (scale >= 1.0) ? 1.0 / scale : 1.0;

  a.begin.resize(out_size);
  a.taps.resize(out_size);
  std::vector<std::vector<double>> rows(static_cast<size_t>(out_size));
  double wmax = 0.0;
  for (int i = 0; i < out_size; ++i) {
    const double center = scale * (i + 0.5);
    // The casts are C truncations of a non-negative value in torch, i.e. floor.
    const int xmin = std::max(static_cast<int>(center - support + 0.5), 0);
    const int xmax = std::min(static_cast<int>(center + support + 0.5), in_size) - xmin;
    std::vector<double> row(static_cast<size_t>(std::max(xmax, 0)));
    double total = 0.0;
    for (int j = 0; j < xmax; ++j) {
      const double v = CubicFilter((j + xmin - center + 0.5) * invscale);
      row[static_cast<size_t>(j)] = v;
      total += v;
    }
    if (total != 0.0) {
      for (double& v : row) v /= total;
    }
    for (double v : row) wmax = std::max(wmax, std::fabs(v));
    a.begin[static_cast<size_t>(i)] = xmin;
    a.taps[static_cast<size_t>(i)] = std::max(xmax, 0);
    a.max_taps = std::max(a.max_taps, a.taps[static_cast<size_t>(i)]);
    rows[static_cast<size_t>(i)] = std::move(row);
  }

  // Fixed-point precision: the largest shift for which every scaled weight still fits in an
  // int16, exactly torch's loop. More precision than the float path needs, but it is the *same*
  // quantization the reference applies, which is what makes the two agree byte for byte.
  int precision = 0;
  while (precision < 22) {
    const int next = static_cast<int>(0.5 + wmax * static_cast<double>(1 << (precision + 1)));
    if (next >= (1 << 15)) break;
    ++precision;
  }
  a.precision = precision;

  a.w.assign(static_cast<size_t>(out_size) * static_cast<size_t>(std::max(a.max_taps, 1)), 0);
  for (int i = 0; i < out_size; ++i) {
    const auto& row = rows[static_cast<size_t>(i)];
    for (size_t j = 0; j < row.size(); ++j) {
      const double scaled = row[j] * static_cast<double>(1 << precision);
      const double rounded = scaled >= 0.0 ? std::floor(scaled + 0.5) : -std::floor(0.5 - scaled);
      a.w[static_cast<size_t>(i) * static_cast<size_t>(a.max_taps) + j] =
          static_cast<int32_t>(rounded);
    }
  }
  return a;
}

// One separable pass over the last (innermost, stride-`channels`) spatial axis of a
// [rows, in_size, channels] uint8 buffer, producing [rows, out_size, channels] uint8. The output
// being uint8 -- not a float intermediate carried into the second pass -- is part of the contract:
// torch's uint8 kernel rounds and clamps between passes, and a float intermediate disagrees with
// it wherever bicubic's negative lobes overshoot, which is every high-contrast edge.
void ResamplePass(const uint8_t* src, int rows, int in_size, int channels, const AxisWeights& a,
                   uint8_t* dst) {
  const int32_t rounding = a.precision > 0 ? (1 << (a.precision - 1)) : 0;
  for (int r = 0; r < rows; ++r) {
    const uint8_t* src_row = src + static_cast<size_t>(r) * in_size * channels;
    uint8_t* dst_row = dst + static_cast<size_t>(r) * a.out_size * channels;
    for (int o = 0; o < a.out_size; ++o) {
      const int begin = a.begin[static_cast<size_t>(o)];
      const int taps = a.taps[static_cast<size_t>(o)];
      const int32_t* wp = a.w.data() + static_cast<size_t>(o) * static_cast<size_t>(a.max_taps);
      for (int c = 0; c < channels; ++c) {
        int64_t acc = rounding;
        for (int j = 0; j < taps; ++j) {
          acc += static_cast<int64_t>(src_row[static_cast<size_t>(begin + j) * channels + c]) * wp[j];
        }
        const int64_t v = acc >> a.precision;
        dst_row[static_cast<size_t>(o) * channels + c] =
            static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
      }
    }
  }
}

// Transpose a [rows, cols, channels] uint8 buffer to [cols, rows, channels], so the vertical pass
// can reuse ResamplePass (which only ever walks the innermost spatial axis).
void TransposeHwc(const uint8_t* src, int rows, int cols, int channels, uint8_t* dst) {
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      const uint8_t* s = src + (static_cast<size_t>(r) * cols + c) * channels;
      uint8_t* d = dst + (static_cast<size_t>(c) * rows + r) * channels;
      for (int k = 0; k < channels; ++k) d[k] = s[k];
    }
  }
}

// One image's patches, appended to `out`. See preprocess.h's point 2: the row order here is the
// whole reason the tower's merger can be a plain reshape.
void PatchifyInto(const DecodedImage& img, const ImageProcessorConfig& cfg, const GridThw& grid,
                   std::vector<float>* out) {
  // The loop nest below walks one frame; a t > 1 grid would size the output for t frames and fill
  // only the first. PreprocessImages only ever builds t == 1 grids, so this is a guard against a
  // future video caller, not a live case.
  if (grid.t != 1) throw std::runtime_error("PatchifyInto: only still images (grid.t == 1) are supported");
  const int patch = cfg.patch_size;
  const int merge = cfg.merge_size;
  const int64_t blocks_h = grid.h / merge;
  const int64_t blocks_w = grid.w / merge;
  const int64_t patch_dim = cfg.PatchDim();
  const size_t base = out->size();
  out->resize(base + static_cast<size_t>(grid.PatchCount() * patch_dim));
  float* dst_all = out->data() + base;

  // The fused rescale+normalize the reference applies: image_mean/std are pre-scaled by
  // 1/rescale_factor (TorchvisionBackend::_fuse_mean_std_and_rescale_factor), so the uint8 value
  // never passes through a [0,1] intermediate -- it is (u8 - 127.5) / 127.5 straight from the
  // byte. Note the DIVISION: tvF.normalize is sub_() then div_(), and multiplying by a
  // precomputed reciprocal instead is off by an fp32 ulp on most pixels (measured: max 1.19e-7),
  // which is enough to lose the bit-exactness this whole path is built for.
  float mean[3], std[3];
  cfg.FusedMeanStd(mean, std);

  int64_t patch_index = 0;
  for (int64_t br = 0; br < blocks_h; ++br) {
    for (int64_t bc = 0; bc < blocks_w; ++bc) {
      for (int64_t in_row = 0; in_row < merge; ++in_row) {
        for (int64_t in_col = 0; in_col < merge; ++in_col, ++patch_index) {
          const int64_t grid_row = br * merge + in_row;
          const int64_t grid_col = bc * merge + in_col;
          float* dst = dst_all + patch_index * patch_dim;
          // Within a patch row the layout is [channel][temporal][patch_y][patch_x]; the temporal
          // axis holds temporal_patch_size identical copies of the single frame (the reference's
          // unsqueeze(6).expand(...) over a still image).
          for (int c = 0; c < 3; ++c) {
            for (int py = 0; py < patch; ++py) {
              const int64_t y = grid_row * patch + py;
              const uint8_t* src_row = img.rgb.data() +
                                       (static_cast<size_t>(y) * img.width +
                                        static_cast<size_t>(grid_col * patch)) * 3;
              for (int px = 0; px < patch; ++px) {
                const float v =
                    (static_cast<float>(src_row[static_cast<size_t>(px) * 3 + c]) - mean[c]) /
                    std[c];
                for (int t = 0; t < cfg.temporal_patch_size; ++t) {
                  dst[((static_cast<int64_t>(c) * cfg.temporal_patch_size + t) * patch + py) * patch + px] = v;
                }
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace

void SmartResize(int64_t height, int64_t width, int64_t factor, int64_t min_pixels,
                  int64_t max_pixels, int64_t* out_height, int64_t* out_width) {
  if (height <= 0 || width <= 0) throw std::runtime_error("smart_resize: image has a zero side");
  const double lo = static_cast<double>(std::min(height, width));
  const double hi = static_cast<double>(std::max(height, width));
  if (hi / lo > 200.0) {
    throw std::runtime_error("smart_resize: absolute aspect ratio must be smaller than 200, got " +
                              std::to_string(hi / lo));
  }
  const double f = static_cast<double>(factor);
  int64_t h_bar = RoundHalfToEven(static_cast<double>(height) / f) * factor;
  int64_t w_bar = RoundHalfToEven(static_cast<double>(width) / f) * factor;
  if (h_bar * w_bar > max_pixels) {
    const double beta =
        std::sqrt(static_cast<double>(height) * static_cast<double>(width) /
                   static_cast<double>(max_pixels));
    h_bar = std::max(factor, static_cast<int64_t>(std::floor(height / beta / f)) * factor);
    w_bar = std::max(factor, static_cast<int64_t>(std::floor(width / beta / f)) * factor);
  } else if (h_bar * w_bar < min_pixels) {
    const double beta = std::sqrt(static_cast<double>(min_pixels) /
                                   (static_cast<double>(height) * static_cast<double>(width)));
    h_bar = static_cast<int64_t>(std::ceil(height * beta / f)) * factor;
    w_bar = static_cast<int64_t>(std::ceil(width * beta / f)) * factor;
  }
  *out_height = h_bar;
  *out_width = w_bar;
}

DecodedImage ResizeU8(const DecodedImage& src, int out_width, int out_height) {
  if (out_width <= 0 || out_height <= 0) throw std::runtime_error("ResizeU8: non-positive target");
  if (out_width == src.width && out_height == src.height) return src;

  constexpr int kChannels = 3;
  const AxisWeights wx = BuildAxisWeights(src.width, out_width);
  const AxisWeights wy = BuildAxisWeights(src.height, out_height);

  // Horizontal first, then vertical -- the order torch's separable kernel uses. Reversing it
  // changes the result (the intermediate is quantized to uint8, so the two passes do not commute).
  std::vector<uint8_t> horiz(static_cast<size_t>(src.height) * out_width * kChannels);
  ResamplePass(src.rgb.data(), src.height, src.width, kChannels, wx, horiz.data());

  std::vector<uint8_t> horiz_t(horiz.size());
  TransposeHwc(horiz.data(), src.height, out_width, kChannels, horiz_t.data());

  std::vector<uint8_t> vert_t(static_cast<size_t>(out_width) * out_height * kChannels);
  ResamplePass(horiz_t.data(), out_width, src.height, kChannels, wy, vert_t.data());

  DecodedImage out;
  out.width = out_width;
  out.height = out_height;
  out.rgb.resize(vert_t.size());
  TransposeHwc(vert_t.data(), out_width, out_height, kChannels, out.rgb.data());
  return out;
}

ImageProcessorConfig MakeImageProcessorConfig(int64_t image_max_pixels) {
  if (image_max_pixels < 0) {
    throw std::runtime_error("MakeImageProcessorConfig: image_max_pixels must be >= 0");
  }
  ImageProcessorConfig cfg;  // this checkpoint's preprocessor_config.json defaults
  if (image_max_pixels > 0) {
    // Never below the floor: smart_resize's min_pixels branch would then fight the cap, upscaling
    // what the cap just downscaled. A cap under the floor means the caller wants the floor.
    cfg.max_pixels = std::max(image_max_pixels, cfg.min_pixels);
  }
  return cfg;
}

PreprocessedImages PreprocessImages(const std::vector<DecodedImage>& images,
                                     const ImageProcessorConfig& cfg) {
  PreprocessedImages out;
  out.patch_dim = cfg.PatchDim();
  out.grid_thw.reserve(images.size());
  for (const DecodedImage& img : images) {
    if (img.rgb.size() != img.PixelCount() * 3) {
      throw std::runtime_error("PreprocessImages: DecodedImage buffer size does not match w*h*3");
    }
    int64_t rh = 0, rw = 0;
    SmartResize(img.height, img.width, cfg.Factor(), cfg.min_pixels, cfg.max_pixels, &rh, &rw);
    const DecodedImage resized = ResizeU8(img, static_cast<int>(rw), static_cast<int>(rh));

    GridThw grid;
    grid.t = 1;
    grid.h = rh / cfg.patch_size;
    grid.w = rw / cfg.patch_size;
    PatchifyInto(resized, cfg, grid, &out.pixel_values);
    out.grid_thw.push_back(grid);
  }
  return out;
}

}  // namespace r4dx::vision
