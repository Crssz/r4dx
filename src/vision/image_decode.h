// src/vision/image_decode.h -- host-side image bytes -> 8-bit RGB pixels, via the vendored
// third_party/stb/stb_image.h (PNG / JPEG / BMP / GIF-first-frame, plus whatever else stb_image
// happens to accept; this header does not gate on the format, stb_image's own sniffing decides).
//
// The RGB conversion has to match what the reference preprocessing pipeline does, not just "look
// right": transformers' Qwen2VLImageProcessorFast sets do_convert_rgb=True, which calls PIL's
// Image.convert("RGB"), and PIL's RGBA->RGB conversion DROPS the alpha channel -- it does NOT
// composite the image over white (or any other) background. stb_image with req_comp=3 does exactly
// the same thing (its 4->3 conversion copies R,G,B and discards A), and replicates the single
// channel of a greyscale image across all three, which is also what PIL does. That agreement is
// why this file passes 3 to stb_image and does no alpha handling of its own; see docs/vision.md
// "Preprocessing" for the measurement behind that claim.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace r4dx::vision {

// 8-bit RGB, row-major, interleaved: pixel (y, x) channel c is at (y * width + x) * 3 + c.
struct DecodedImage {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> rgb;

  size_t PixelCount() const { return static_cast<size_t>(width) * static_cast<size_t>(height); }
};

// Throws std::runtime_error with stb_image's own failure reason if the bytes are not a decodable
// image. A multi-frame GIF decodes to its first frame (stb_image's single-image entry point).
DecodedImage DecodeImageBytes(const uint8_t* data, size_t size);

DecodedImage DecodeImageFile(const std::string& path);

}  // namespace r4dx::vision
