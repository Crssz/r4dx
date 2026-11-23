#include "inkling2_vision.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace r4dx::vision {

Inkling2Preprocessed PreprocessInkling2Image(const DecodedImage& img, const Inkling2ImageConfig& cfg) {
  if (img.width <= 0 || img.height <= 0 ||
      img.rgb.size() != static_cast<size_t>(img.width) * static_cast<size_t>(img.height) * 3) {
    throw std::invalid_argument("PreprocessInkling2Image: empty or malformed RGB image");
  }

  double h = static_cast<double>(img.height);
  double w = static_cast<double>(img.width);
  const int64_t max_side = std::clamp<int64_t>(cfg.max_side, kInkling2Factor, 4096);
  const int64_t max_pixels = std::max<int64_t>(cfg.max_pixels, static_cast<int64_t>(kInkling2Factor) * kInkling2Factor);

  // Downscale if exceeding max_side or max_pixels, preserving aspect ratio.
  double scale = 1.0;
  if (std::max(h, w) > static_cast<double>(max_side)) {
    scale = std::min(scale, static_cast<double>(max_side) / std::max(h, w));
  }
  if (h * w * scale * scale > static_cast<double>(max_pixels)) {
    scale = std::min(scale, std::sqrt(static_cast<double>(max_pixels) / (h * w)));
  }

  int64_t rh = std::max<int64_t>(1, static_cast<int64_t>(std::llround(h * scale)));
  int64_t rw = std::max<int64_t>(1, static_cast<int64_t>(std::llround(w * scale)));

  const DecodedImage resized_img =
      (rh == img.height && rw == img.width) ? img : ResizeU8(img, static_cast<int>(rw), static_cast<int>(rh));
  const std::vector<uint8_t>& resized = resized_img.rgb;

  // Pad on the right and bottom to multiples of 40 (native_patch_size * adapter_merge_size).
  const int64_t ph = ((rh + kInkling2Factor - 1) / kInkling2Factor) * kInkling2Factor;
  const int64_t pw = ((rw + kInkling2Factor - 1) / kInkling2Factor) * kInkling2Factor;
  const int64_t gh = ph / kInkling2NativePatch;
  const int64_t gw = pw / kInkling2NativePatch;

  Inkling2Preprocessed out;
  out.grid_h = gh;
  out.grid_w = gw;
  out.resized_h = rh;
  out.resized_w = rw;
  out.pixel_values.assign(static_cast<size_t>(gh * gw) * kInkling2PatchDim, 0.0f);

  // Normalize real pixels to [-1, 1] via (u8 / 255 - 0.5) / 0.5 = u8 * (2/255) - 1; padded pixels stay 0.0f.
  // Slice into native 20x20x3 HWC patches in row-major native grid order (gy, gx).
  constexpr float kScale = 2.0f / 255.0f;
  for (int64_t gy = 0; gy < gh; ++gy) {
    for (int64_t gx = 0; gx < gw; ++gx) {
      float* dst = out.pixel_values.data() + static_cast<size_t>(gy * gw + gx) * kInkling2PatchDim;
      const int64_t y0 = gy * kInkling2NativePatch;
      const int64_t x0 = gx * kInkling2NativePatch;
      for (int64_t dy = 0; dy < kInkling2NativePatch; ++dy) {
        const int64_t y = y0 + dy;
        if (y >= rh) break;
        const uint8_t* src_row = resized.data() + static_cast<size_t>(y * rw) * 3;
        float* dst_row = dst + static_cast<size_t>(dy * kInkling2NativePatch) * 3;
        const int64_t x_lim = std::min<int64_t>(kInkling2NativePatch, rw - x0);
        for (int64_t dx = 0; dx < x_lim; ++dx) {
          const uint8_t* px = src_row + static_cast<size_t>(x0 + dx) * 3;
          dst_row[dx * 3 + 0] = static_cast<float>(px[0]) * kScale - 1.0f;
          dst_row[dx * 3 + 1] = static_cast<float>(px[1]) * kScale - 1.0f;
          dst_row[dx * 3 + 2] = static_cast<float>(px[2]) * kScale - 1.0f;
        }
      }
    }
  }
  return out;
}

}  // namespace r4dx::vision
