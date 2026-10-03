// tests/vision/test_gemma_preprocess_golden.cpp -- r4dx::vision::PreprocessGemmaImage against
// transformers' Gemma4UnifiedImageProcessor, BIT for bit (pixel_values as float32 patches, positions as
// int32), on the synthetic images tools/reference/gemma/vision_preproc_golden.py writes (noise and gradient
// images, up- and down-scaled, a one-row strip, a no-resize image, the 70 / 140 / 560 budgets). Also runs the
// path through PreprocessImages (the server's entry) with ImageProcessorConfig::gemma set.
//
// Exits 77 (SKIPPED) when the golden directory is absent.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "gemma_test_util.h"
#include "gemma_vision.h"
#include "preprocess.h"

using namespace r4dx::vision;

int main() {
  const std::string dir = gemma_test::GoldenDir() + "/vision_preproc";
  std::vector<uint8_t> manifest;
  if (!gemma_test::ReadFile(dir + "/case0.rgb", &manifest)) {
    return gemma_test::Skip(dir, "python tools/reference/gemma/vision_preproc_golden.py --check");
  }
  int cases = 0;
  for (int i = 0;; ++i) {
    std::vector<uint8_t> rgb, pix, pos;
    if (!gemma_test::ReadFile(dir + "/case" + std::to_string(i) + ".rgb", &rgb)) break;
    GCHECK(gemma_test::ReadFile(dir + "/case" + std::to_string(i) + ".pix", &pix));
    GCHECK(gemma_test::ReadFile(dir + "/case" + std::to_string(i) + ".pos", &pos));
    int32_t w = 0, h = 0;
    std::memcpy(&w, rgb.data(), 4);
    std::memcpy(&h, rgb.data() + 4, 4);
    DecodedImage img;
    img.width = w;
    img.height = h;
    img.rgb.assign(rgb.begin() + 8, rgb.end());
    GCHECK(img.rgb.size() == static_cast<size_t>(w) * h * 3);
    int32_t mst = 0, n = 0;
    std::memcpy(&mst, pix.data(), 4);
    std::memcpy(&n, pix.data() + 4, 4);
    GCHECK(pix.size() == 8 + static_cast<size_t>(n) * kGemmaPatchDim * 4);
    GCHECK(pos.size() == static_cast<size_t>(n) * 2 * 4);

    GemmaImageConfig cfg;
    cfg.max_soft_tokens = mst;
    const GemmaPreprocessed g = PreprocessGemmaImage(img, cfg);
    GCHECK(g.n == n);
    GCHECK(g.pixel_values.size() == static_cast<size_t>(n) * kGemmaPatchDim);
    GCHECK(g.positions.size() == static_cast<size_t>(n) * 2);
    size_t bad = 0, first_bad = 0;
    const uint8_t* want = pix.data() + 8;
    for (size_t k = 0; k < g.pixel_values.size() && k * 4 + 4 <= pix.size() - 8; ++k) {
      uint32_t a, b;
      std::memcpy(&a, &g.pixel_values[k], 4);
      std::memcpy(&b, want + k * 4, 4);
      if (a != b) {
        if (bad == 0) first_bad = k;
        ++bad;
      }
    }
    if (bad != 0) {
      std::fprintf(stderr, "  case %d (%dx%d): %zu pixel values differ, first at %zu (patch %zu)\n", i, h, w, bad,
                   first_bad, first_bad / kGemmaPatchDim);
    }
    GCHECK(bad == 0);
    GCHECK(std::memcmp(g.positions.data(), pos.data(), pos.size()) == 0);

    // The server entry point: same tensors, and a grid whose PatchCount() == MergedTokenCount(1) == n.
    ImageProcessorConfig pc;
    pc.gemma = true;
    pc.gemma_soft_tokens = mst;
    const PreprocessedImages pre = PreprocessImages({img}, pc);
    GCHECK(pre.patch_dim == kGemmaPatchDim);
    GCHECK(pre.grid_thw.size() == 1 && pre.grid_thw[0].t == 1 && pre.grid_thw[0].PatchCount() == n &&
           pre.grid_thw[0].MergedTokenCount(1) == n);
    GCHECK(pre.pixel_values == g.pixel_values);
    ++cases;
    std::printf("case %d: %dx%d -> %lldx%lld, %d tokens (budget %d): %s\n", i, h, w,
                static_cast<long long>(g.resized_h), static_cast<long long>(g.resized_w), n, mst,
                bad == 0 ? "bit-identical" : "MISMATCH");
  }
  GCHECK(cases >= 11);
  if (gemma_test::g_failures != 0) {
    std::fprintf(stderr, "test_gemma_preprocess_golden: %d failure(s)\n", gemma_test::g_failures);
    return 1;
  }
  std::printf("test_gemma_preprocess_golden: OK (%d cases)\n", cases);
  return 0;
}
