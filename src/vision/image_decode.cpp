#include "image_decode.h"

#include <cstdint>
#include <cstdio>
#include <stdexcept>

// The one translation unit in r4dx that instantiates stb_image. STBI_ONLY_* keeps the formats to
// the ones a chat client actually attaches (Unsloth Studio's accept list is
// image/jpeg,image/png,image/webp,image/gif -- stb_image has no WEBP decoder, so a WEBP upload is
// rejected with stb's own "unknown image type" message rather than silently mis-decoded), and
// STBI_NO_STDIO is deliberately NOT set: DecodeImageFile below uses stbi_load for the unit tests
// and for r4dx-cli's eventual --image flag.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_GIF
#include "stb/stb_image.h"

namespace r4dx::vision {
namespace {

DecodedImage FromStbBuffer(unsigned char* pixels, int w, int h, const char* what) {
  if (pixels == nullptr) {
    const char* reason = stbi_failure_reason();
    throw std::runtime_error(std::string("image decode failed (") + what + "): " +
                              (reason != nullptr ? reason : "unknown"));
  }
  if (w <= 0 || h <= 0) {
    stbi_image_free(pixels);
    throw std::runtime_error("image decode produced an empty image");
  }
  DecodedImage img;
  img.width = w;
  img.height = h;
  const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h) * 3;
  img.rgb.assign(pixels, pixels + n);
  stbi_image_free(pixels);
  return img;
}

}  // namespace

DecodedImage DecodeImageBytes(const uint8_t* data, size_t size) {
  // stb_image's memory entry point takes an int length; a >2 GB attachment would wrap it into a
  // negative and read out of bounds rather than fail.
  if (size == 0 || size > static_cast<size_t>(INT32_MAX)) {
    throw std::runtime_error("image decode failed: buffer is empty or larger than 2 GiB");
  }
  int w = 0, h = 0, channels_in_file = 0;
  // req_comp=3: see the header comment -- alpha is dropped, greyscale is replicated, matching
  // PIL's Image.convert("RGB") which is what the reference processor applies.
  unsigned char* pixels = stbi_load_from_memory(data, static_cast<int>(size), &w, &h,
                                                 &channels_in_file, 3);
  return FromStbBuffer(pixels, w, h, "memory");
}

DecodedImage DecodeImageFile(const std::string& path) {
  int w = 0, h = 0, channels_in_file = 0;
  unsigned char* pixels = stbi_load(path.c_str(), &w, &h, &channels_in_file, 3);
  return FromStbBuffer(pixels, w, h, path.c_str());
}

}  // namespace r4dx::vision
