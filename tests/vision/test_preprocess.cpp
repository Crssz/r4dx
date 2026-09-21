// tests/vision/test_preprocess.cpp -- r4dx::vision::PreprocessImages against the REAL
// Qwen2VLImageProcessorFast output (tools/reference/vision_golden.py's `pixel_values`), for all
// three golden cases: the 448x448 square image (grid 28x28, smart_resize is the identity), the
// 613x409 non-square one (-> 608x416, grid 26x38: width down-sampled, height up-sampled, h != w),
// and both together in one call (the concatenated [1772, 1536] tensor a two-image request emits).
//
// The assertion is EQUALITY, not a tolerance. The reference pipeline is integer-deterministic --
// PIL RGB conversion, smart_resize, torch's uint8 antialias resampler, then one fp32 affine per
// byte -- so any nonzero difference here is a real disagreement, not accumulated float noise. The
// test still reports max-abs/mean-abs error rather than just pass/fail, because those numbers are
// what a future change to the resampler has to be judged against (docs/vision.md "Resampling"
// records them).
//
// Four further groups sit alongside the three whole-pipeline cases, each pinning a claim the
// pipeline comparison alone would not localize: PIL's RGB conversion (alpha dropped, greyscale
// replicated), the resampler on random noise at four size pairs the golden photographs never
// reach, the BMP/GIF/JPEG decode paths, and -- CPU-only, no golden needed -- the smart_resize
// cases neither golden image hits: the min_pixels floor for a tiny image, the max_pixels ceiling
// for a huge one, Python's round-half-to-even on an exact .5 ratio, and the >200 aspect-ratio
// rejection.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "image_decode.h"
#include "preprocess.h"
#include "vision_test_common.h"

using namespace r4dx_vision_test;
using r4dx::vision::DecodedImage;
using r4dx::vision::GridThw;
using r4dx::vision::ImageProcessorConfig;
using r4dx::vision::MakeImageProcessorConfig;
using r4dx::vision::PreprocessedImages;
using r4dx::vision::PreprocessImages;
using r4dx::vision::SmartResize;

namespace {

// R4DX_SOURCE_DIR is defined by tests/CMakeLists.txt from ${CMAKE_SOURCE_DIR}.
const char* kGoldenDir = R4DX_SOURCE_DIR "/tools/reference/golden_out";

std::string GoldenPath(const std::string& name) { return std::string(kGoldenDir) + "/" + name; }

// One golden case: decode the exact PNG(s) vision_golden.py saved, run them through the real
// preprocessing path, and demand byte-identical patches.
void CheckCase(const std::string& golden_file, const std::vector<std::string>& image_files,
                const std::vector<GridThw>& expected_grids) {
  const r4dx_convert::SafetensorsReader golden(
      r4dx_convert::Utf8ToWide(GoldenPath(golden_file)));

  std::vector<DecodedImage> images;
  for (const std::string& f : image_files) {
    images.push_back(r4dx::vision::DecodeImageFile(GoldenPath(f)));
  }

  const ImageProcessorConfig cfg;
  const auto got = PreprocessImages(images, cfg);

  CHECK(got.grid_thw.size() == expected_grids.size());
  const auto grid_ref = ReadInt(golden, "image_grid_thw");
  CHECK(grid_ref.size() == expected_grids.size() * 3);
  for (size_t i = 0; i < got.grid_thw.size() && i * 3 + 2 < grid_ref.size(); ++i) {
    CHECK(got.grid_thw[i].t == grid_ref[i * 3 + 0]);
    CHECK(got.grid_thw[i].h == grid_ref[i * 3 + 1]);
    CHECK(got.grid_thw[i].w == grid_ref[i * 3 + 2]);
    CHECK(got.grid_thw[i].h == expected_grids[i].h);
    CHECK(got.grid_thw[i].w == expected_grids[i].w);
  }

  const auto ref = ReadFloat(golden, "pixel_values");
  const auto& meta = golden.Meta("pixel_values");
  CHECK(meta.shape.size() == 2 && meta.shape[1] == cfg.PatchDim());
  if (got.pixel_values.size() != ref.size()) {
    std::fprintf(stderr, "%s: pixel_values size %zu != golden %zu -- cannot compare\n",
                 golden_file.c_str(), got.pixel_values.size(), ref.size());
    ++g_failures;
    return;
  }

  const ErrorStats s = CompareFloat(got.pixel_values, ref);
  std::printf("[test_preprocess] %-34s patches=%lld max_abs=%.9g mean_abs=%.9g differing=%lld/%lld\n",
              golden_file.c_str(), static_cast<long long>(got.TotalPatches()), s.max_abs,
              s.mean_abs, static_cast<long long>(s.num_differing), static_cast<long long>(s.count));
  if (s.num_differing != 0) {
    const int64_t i = s.argmax;
    std::fprintf(stderr,
                 "  first/worst disagreement at flat index %lld (patch %lld, dim %lld): got %.9g "
                 "golden %.9g\n",
                 static_cast<long long>(i), static_cast<long long>(i / cfg.PatchDim()),
                 static_cast<long long>(i % cfg.PatchDim()),
                 static_cast<double>(got.pixel_values[static_cast<size_t>(i)]),
                 static_cast<double>(ref[static_cast<size_t>(i)]));
  }
  CHECK(s.max_abs == 0.0);
}

// The 2x2 spatial-merge-block-major patch order is what lets the tower's merger be a plain view();
// this checks it directly against the reference tensor rather than only through the whole-tensor
// equality above, so an ordering bug reports as an ordering bug. Patch p of a grid with
// blocks_w = w/2 covers grid cell (row, col) with
//   row = (p / (4*blocks_w))*2 + (p/2)%2,  col = ((p/4)%blocks_w)*2 + p%2.
// Two patches that share a 2x2 block must therefore be adjacent in the tensor.
void CheckPatchOrderIsBlockMajorNotRaster() {
  const r4dx_convert::SafetensorsReader golden(
      r4dx_convert::Utf8ToWide(GoldenPath("vision_tower_nonsquare.safetensors")));
  const auto ref = ReadFloat(golden, "pixel_values");
  const ImageProcessorConfig cfg;
  const DecodedImage img = r4dx::vision::DecodeImageFile(GoldenPath("vision_test_image_nonsquare.png"));
  int64_t rh = 0, rw = 0;
  SmartResize(img.height, img.width, cfg.Factor(), cfg.min_pixels, cfg.max_pixels, &rh, &rw);
  const DecodedImage resized = r4dx::vision::ResizeU8(img, static_cast<int>(rw), static_cast<int>(rh));

  const int64_t grid_w = rw / cfg.patch_size;
  const int64_t blocks_w = grid_w / cfg.merge_size;
  const int64_t patch_dim = cfg.PatchDim();
  float mean[3], std[3];
  cfg.FusedMeanStd(mean, std);

  // Spot-check a handful of patches spread across the image, reconstructing the expected top-left
  // pixel of each from the resized image independently of PreprocessImages' own loop nest.
  const int64_t num_patches = static_cast<int64_t>(ref.size()) / patch_dim;
  int checked = 0;
  for (int64_t p = 0; p < num_patches; p += 37) {
    const int64_t row = (p / (4 * blocks_w)) * 2 + (p / 2) % 2;
    const int64_t col = ((p / 4) % blocks_w) * 2 + p % 2;
    const int64_t y = row * cfg.patch_size;
    const int64_t x = col * cfg.patch_size;
    for (int c = 0; c < 3; ++c) {
      const float expected =
          (static_cast<float>(resized.rgb[(static_cast<size_t>(y) * resized.width + x) * 3 + c]) -
           mean[c]) / std[c];
      // dim of (channel c, temporal 0, py 0, px 0) is c * 2 * 16 * 16.
      CHECK(ref[static_cast<size_t>(p * patch_dim + c * 2 * 16 * 16)] == expected);
    }
    ++checked;
  }
  std::printf("[test_preprocess] block-major patch order verified on %d sampled patches\n", checked);
}

// The single frame is duplicated across temporal_patch_size, so the t=0 and t=1 halves of every
// channel block must be identical -- a still image never carries two distinct frames.
void CheckTemporalDuplication() {
  const r4dx_convert::SafetensorsReader golden(
      r4dx_convert::Utf8ToWide(GoldenPath("vision_tower.safetensors")));
  const auto ref = ReadFloat(golden, "pixel_values");
  const ImageProcessorConfig cfg;
  const int64_t patch_dim = cfg.PatchDim();
  const int64_t plane = static_cast<int64_t>(cfg.patch_size) * cfg.patch_size;
  const int64_t num_patches = static_cast<int64_t>(ref.size()) / patch_dim;
  int64_t mismatches = 0;
  for (int64_t p = 0; p < num_patches; p += 11) {
    for (int c = 0; c < 3; ++c) {
      for (int64_t i = 0; i < plane; ++i) {
        const size_t t0 = static_cast<size_t>(p * patch_dim + (c * 2 + 0) * plane + i);
        const size_t t1 = static_cast<size_t>(p * patch_dim + (c * 2 + 1) * plane + i);
        if (ref[t0] != ref[t1]) ++mismatches;
      }
    }
  }
  CHECK(mismatches == 0);
}

// The one claim in src/vision/image_decode.h that is a reading of someone else's source rather
// than arithmetic: PIL's Image.convert("RGB") DROPS an alpha channel (it does not composite the
// image over white or any other background), and replicates a greyscale channel across all three.
// vision_golden.py saves an RGBA PNG whose alpha sweeps the full 0..255 range -- including fully
// transparent columns, whose RGB would be unrecoverable if anything composited them -- plus a
// greyscale PNG, and dumps PIL's own output for each. stb_image with req_comp=3 must agree byte
// for byte, or every transparent image r4dx is handed is silently wrong.
void CheckRgbConversionMatchesPil() {
  const r4dx_convert::SafetensorsReader golden(
      r4dx_convert::Utf8ToWide(GoldenPath("vision_tower.safetensors")));
  const struct {
    const char* png;
    const char* tensor;
  } cases[] = {{"vision_test_image_rgba.png", "rgba_pil_rgb"},
                {"vision_test_image_grey.png", "grey_pil_rgb"}};
  for (const auto& c : cases) {
    if (!FileExists(GoldenPath(c.png)) || !golden.Has(c.tensor)) {
      std::fprintf(stderr, "[test_preprocess] %s missing from the golden -- regenerate with "
                            "tools/reference/vision_golden.py\n", c.tensor);
      ++g_failures;
      continue;
    }
    const DecodedImage img = r4dx::vision::DecodeImageFile(GoldenPath(c.png));
    const auto ref = ReadU8(golden, c.tensor);
    CHECK(img.rgb.size() == ref.size());
    int64_t bad = 0;
    for (size_t i = 0; i < img.rgb.size() && i < ref.size(); ++i) {
      if (img.rgb[i] != ref[i]) ++bad;
    }
    CHECK(bad == 0);
    std::printf("[test_preprocess] %-34s %dx%d differing_bytes=%lld\n", c.png, img.width,
                img.height, static_cast<long long>(bad));
  }
}

// The resampler on its own, against the reference's exact `tvF.resize(uint8, BICUBIC,
// antialias=True)` call, on four random-noise size pairs (both axes down, the real smart_resize
// pair, both axes up, a strong downscale). Noise is the worst case for any rounding or kernel
// disagreement -- the four properties docs/vision.md "Resampling" lists (Pillow's a = -0.5,
// support widened by the downscale factor, int16 fixed-point weights, horizontal-first passes
// through a uint8 intermediate) each cost up to ~25 of 255 on a fifth to nine tenths of the bytes
// when reproduced wrongly, and none of it is visible in a smooth gradient. The two golden
// photographs exercise exactly one resize between them and no upscale at all, which is why this
// case exists separately.
void CheckResamplerOnNoise() {
  const r4dx_convert::SafetensorsReader golden(
      r4dx_convert::Utf8ToWide(GoldenPath("vision_tower.safetensors")));
  int64_t total = 0;
  for (int i = 0;; ++i) {
    const std::string tensor = "resize_noise_" + std::to_string(i) + "_out";
    const std::string png = "vision_resize_noise_" + std::to_string(i) + ".png";
    if (!golden.Has(tensor)) {
      CHECK(i > 0);  // none at all means the golden predates this fixture -- regenerate it
      break;
    }
    if (!FileExists(GoldenPath(png))) {
      std::fprintf(stderr, "[test_preprocess] %s is in the golden but %s is missing\n",
                   tensor.c_str(), png.c_str());
      ++g_failures;
      continue;
    }
    // The target size comes from the golden tensor's own [out_h, out_w, 3] shape, so the test
    // cannot disagree with the reference about what was asked for.
    const auto& meta = golden.Meta(tensor);
    CHECK(meta.shape.size() == 3 && meta.shape[2] == 3);
    if (meta.shape.size() != 3) return;
    const int out_h = static_cast<int>(meta.shape[0]);
    const int out_w = static_cast<int>(meta.shape[1]);

    const DecodedImage src = r4dx::vision::DecodeImageFile(GoldenPath(png));
    const DecodedImage got = r4dx::vision::ResizeU8(src, out_w, out_h);
    const auto ref = ReadU8(golden, tensor);
    CHECK(got.rgb.size() == ref.size());
    int64_t bad = 0, max_abs = 0;
    for (size_t k = 0; k < got.rgb.size() && k < ref.size(); ++k) {
      const int64_t d = std::abs(static_cast<int>(got.rgb[k]) - static_cast<int>(ref[k]));
      if (d != 0) ++bad;
      if (d > max_abs) max_abs = d;
    }
    CHECK(bad == 0);
    total += static_cast<int64_t>(ref.size());
    std::printf("[test_preprocess] resize %4dx%-4d -> %4dx%-4d max_abs=%lld differing=%lld/%lld\n",
                src.width, src.height, out_w, out_h, static_cast<long long>(max_abs),
                static_cast<long long>(bad), static_cast<long long>(ref.size()));
  }
  std::printf("[test_preprocess] resampler checked over %lld bytes of random noise\n",
              static_cast<long long>(total));
}

// src/vision/image_decode.h claims PNG/JPEG/BMP/GIF-first-frame; PNG is covered everywhere above,
// these are the other three. BMP and GIF are lossless given the file (a GIF's 256-colour palette
// reads identically in both decoders), so they must match PIL byte for byte; JPEG cannot, because
// stb_image's IDCT is not libjpeg's -- it gets a small bound and its measured error is printed
// rather than an equality assertion that would be a lie.
void CheckDecodesOtherFormats() {
  const r4dx_convert::SafetensorsReader golden(
      r4dx_convert::Utf8ToWide(GoldenPath("vision_tower.safetensors")));
  const struct {
    const char* file;
    const char* tensor;
    int tolerance;
  } cases[] = {{"vision_test_image_format.bmp", "bmp_pil_rgb", 0},
                {"vision_test_image_format.gif", "gif_pil_rgb", 0},
                {"vision_test_image_format.jpg", "jpeg_pil_rgb", 3}};
  for (const auto& c : cases) {
    if (!golden.Has(c.tensor) || !FileExists(GoldenPath(c.file))) {
      std::fprintf(stderr, "[test_preprocess] %s missing from the golden -- regenerate with "
                            "tools/reference/vision_golden.py\n", c.tensor);
      ++g_failures;
      continue;
    }
    const DecodedImage img = r4dx::vision::DecodeImageFile(GoldenPath(c.file));
    const auto ref = ReadU8(golden, c.tensor);
    CHECK(img.rgb.size() == ref.size());
    int64_t bad = 0, max_abs = 0;
    for (size_t i = 0; i < img.rgb.size() && i < ref.size(); ++i) {
      const int64_t d = std::abs(static_cast<int>(img.rgb[i]) - static_cast<int>(ref[i]));
      if (d != 0) ++bad;
      if (d > max_abs) max_abs = d;
    }
    CHECK(max_abs <= c.tolerance);
    std::printf("[test_preprocess] %-34s %dx%d max_abs=%lld differing_bytes=%lld (tol %d)\n",
                c.file, img.width, img.height, static_cast<long long>(max_abs),
                static_cast<long long>(bad), c.tolerance);
  }
}

// --image-max-pixels (docs/vision.md "Large images"): the cap must DOWNSIZE a big image through
// smart_resize's own ceiling branch, never reject it, and a whole preprocessing run under the cap
// must produce a grid whose patch count really is bounded.
void CheckImageMaxPixelsCap() {
  const ImageProcessorConfig def = MakeImageProcessorConfig(0);
  CHECK(def.max_pixels == 16777216);  // 0 keeps the checkpoint's own preprocessor_config ceiling

  const ImageProcessorConfig capped = MakeImageProcessorConfig(1048576);
  CHECK(capped.max_pixels == 1048576);
  // A cap under the min_pixels FLOOR would make the two branches fight (the floor would re-upscale
  // what the cap just downscaled), so it clamps up to the floor instead.
  CHECK(MakeImageProcessorConfig(1024).max_pixels == def.min_pixels);

  // A 2048x2048 image (16384 patches uncapped) under the 1024x1024 default cap.
  DecodedImage big;
  big.width = 2048;
  big.height = 2048;
  big.rgb.assign(static_cast<size_t>(big.width) * big.height * 3, 0);
  for (size_t i = 0; i < big.rgb.size(); ++i) big.rgb[i] = static_cast<uint8_t>(i * 7);

  const PreprocessedImages uncapped_out = PreprocessImages({big}, def);
  const PreprocessedImages capped_out = PreprocessImages({big}, capped);
  CHECK(uncapped_out.grid_thw.size() == 1 && capped_out.grid_thw.size() == 1);
  const GridThw& gu = uncapped_out.grid_thw[0];
  const GridThw& gc = capped_out.grid_thw[0];
  std::printf("[image-max-pixels] 2048x2048: uncapped grid %lldx%lld (%lld patches) -> capped "
              "%lldx%lld (%lld patches)\n",
              static_cast<long long>(gu.h), static_cast<long long>(gu.w),
              static_cast<long long>(gu.PatchCount()), static_cast<long long>(gc.h),
              static_cast<long long>(gc.w), static_cast<long long>(gc.PatchCount()));
  CHECK(gu.h == 128 && gu.w == 128);   // 2048/16, untouched: 4M pixels is under the 16M ceiling
  CHECK(gc.h == 64 && gc.w == 64);     // downsized to 1024x1024, i.e. exactly the cap
  CHECK(gc.h * gc.w * 16 * 16 <= capped.max_pixels);
  CHECK(gc.h % capped.merge_size == 0 && gc.w % capped.merge_size == 0);
  // Downsized, not rejected: the tensor really is there, and it is the capped size.
  CHECK(capped_out.TotalPatches() == gc.PatchCount());
  CHECK(capped_out.pixel_values.size() ==
        static_cast<size_t>(gc.PatchCount() * capped.PatchDim()));
}

void CheckSmartResizeEdgeCases() {
  int64_t h = 0, w = 0;
  const int64_t factor = 32, lo = 65536, hi = 16777216;

  // The two golden images, as a pin on the sizes the rest of this test assumes.
  SmartResize(448, 448, factor, lo, hi, &h, &w);
  CHECK(h == 448 && w == 448);
  SmartResize(409, 613, factor, lo, hi, &h, &w);
  CHECK(h == 416 && w == 608);

  // Below the min_pixels floor: scaled UP to at least 65536 total pixels, each side ceil'd to a
  // multiple of factor.
  SmartResize(37, 41, factor, lo, hi, &h, &w);
  CHECK(h == 256 && w == 288);
  CHECK(h % factor == 0 && w % factor == 0);
  CHECK(h * w >= lo);

  // Above the max_pixels ceiling: scaled DOWN, each side floor'd to a multiple of factor.
  SmartResize(8000, 6000, factor, lo, 4096 * 4096, &h, &w);
  CHECK(h % factor == 0 && w % factor == 0);
  CHECK(h * w <= 4096 * 4096);

  // Exact .5 ratios, checked with a pixel budget wide enough that neither the floor nor the
  // ceiling branch fires, so only the rounding is under test. Python's round() is half-to-even:
  // 48/32 = 1.5 rounds UP to 2 (2 is even) while 80/32 = 2.5 rounds DOWN to 2 -- both give 64.
  // A floor(x + 0.5) implementation gets the second one wrong (3 -> 96).
  SmartResize(48, 288, factor, 1, hi, &h, &w);
  CHECK(h == 64 && w == 288);
  SmartResize(80, 288, factor, 1, hi, &h, &w);
  CHECK(h == 64 && w == 288);

  bool threw = false;
  try {
    SmartResize(10, 2500, factor, lo, hi, &h, &w);  // aspect ratio 250 > 200
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);
}

}  // namespace

int main() {
  const std::vector<std::string> needed = {
      "vision_tower.safetensors", "vision_tower_nonsquare.safetensors",
      "vision_tower_two_image.safetensors", "vision_test_image.png",
      "vision_test_image_nonsquare.png"};
  for (const std::string& f : needed) {
    if (!FileExists(GoldenPath(f))) return SkipMissing(GoldenPath(f));
  }

  try {
    CheckCase("vision_tower.safetensors", {"vision_test_image.png"}, {GridThw{1, 28, 28}});
    CheckCase("vision_tower_nonsquare.safetensors", {"vision_test_image_nonsquare.png"},
              {GridThw{1, 26, 38}});
    CheckCase("vision_tower_two_image.safetensors",
              {"vision_test_image.png", "vision_test_image_nonsquare.png"},
              {GridThw{1, 28, 28}, GridThw{1, 26, 38}});
    CheckPatchOrderIsBlockMajorNotRaster();
    CheckTemporalDuplication();
    CheckRgbConversionMatchesPil();
    CheckResamplerOnNoise();
    CheckDecodesOtherFormats();
    CheckSmartResizeEdgeCases();
    CheckImageMaxPixelsCap();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "test_preprocess: unexpected exception: %s\n", e.what());
    return 1;
  }

  if (g_failures != 0) {
    std::fprintf(stderr, "test_preprocess: %d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("test_preprocess: OK\n");
  return 0;
}

