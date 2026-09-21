// tests/vision/test_vision_index.cpp -- the grid_thw-derived index tables in src/vision:
//
//   cu_seqlens                 exact match vs the reference (1 segment for a single image, 2 for
//                              the two-image case -- the case the previous pass never exercised).
//   vision rope (h, w) ids     exact match, including the 2x2 block-major ordering.
//   pos-embed interpolation    exact match on both the 4 gather indices and the 4 weights per
//                              patch, then the position embedding they produce: gathered from the
//                              real learned [2304, 1152] table (dumped into the golden so this
//                              test needs no converted container) and compared against the
//                              reference's own `pos_embeds`.
//   vision rope cos/sin        compared against the table the reference actually handed the
//                              encoder blocks.
//
// The float comparisons carry a tolerance only where the reference's own arithmetic makes bit
// equality unattainable: `pos_embeds` is a 4-term fp32 sum whose association order torch does not
// document, and cos/sin/inv_freq come out of a different libm than this binary's. Everything
// integer is asserted exactly.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "preprocess.h"
#include "vision_index.h"
#include "vision_test_common.h"

using namespace r4dx_vision_test;
using r4dx::vision::BuildCuSeqlens;
using r4dx::vision::BuildPosEmbedInterpolation;
using r4dx::vision::BuildVisionRopeCosSin;
using r4dx::vision::BuildVisionRopePositionIds;
using r4dx::vision::GridThw;

namespace {

const char* kGoldenDir = R4DX_SOURCE_DIR "/tools/reference/golden_out";

std::string GoldenPath(const std::string& name) { return std::string(kGoldenDir) + "/" + name; }

// docs/vision.md "Model facts", from the real checkpoint's vision_config.
constexpr int kNumGridPerSide = 48;  // sqrt(num_position_embeddings = 2304)
constexpr int kMergeSize = 2;
constexpr int kHiddenSize = 1152;
constexpr int kHeadDim = 72;
constexpr double kVisionRopeTheta = 10000.0;

// A 4-term fp32 sum in an unspecified association order; the reference's own
// fp32_elementwise_rel_err budget (1e-4) is far looser than what is actually observed here.
constexpr double kPosEmbedAbsTol = 1e-5;
// inv_freq is theta^(-2i/spatial_dim), evaluated in fp32 by the reference and in double-then-
// -rounded here; ~1e-7 of relative disagreement in the frequency, multiplied by a position index
// up to ~40, lands a few 1e-6 off in the angle. 2e-5 is that with headroom -- still three orders
// of magnitude tighter than the 0.047 a bf16-rounded inv_freq would cost (see docs/vision.md
// "Rope"), so this bound still catches the mistake it exists to catch.
constexpr double kRopeAbsTol = 2e-5;

std::vector<GridThw> GridsFromGolden(const r4dx_convert::SafetensorsReader& g) {
  const auto raw = ReadInt(g, "image_grid_thw");
  std::vector<GridThw> grids;
  for (size_t i = 0; i + 2 < raw.size(); i += 3) grids.push_back(GridThw{raw[i], raw[i + 1], raw[i + 2]});
  return grids;
}

void CheckCase(const std::string& golden_file, const std::vector<float>& pos_embed_table) {
  const r4dx_convert::SafetensorsReader golden(r4dx_convert::Utf8ToWide(GoldenPath(golden_file)));
  const std::vector<GridThw> grids = GridsFromGolden(golden);

  // ---- cu_seqlens -----------------------------------------------------------------------
  const auto cu_ref = ReadInt(golden, "cu_seqlens");
  const auto cu_got = BuildCuSeqlens(grids);
  CHECK(cu_got.size() == cu_ref.size());
  int64_t cu_bad = 0;
  for (size_t i = 0; i < cu_got.size() && i < cu_ref.size(); ++i) {
    if (cu_got[i] != cu_ref[i]) ++cu_bad;
  }
  CHECK(cu_bad == 0);

  // ---- vision rope position ids --------------------------------------------------------
  const auto pos_ref = ReadInt(golden, "vision_position_ids");
  const auto pos_got = BuildVisionRopePositionIds(grids, kMergeSize);
  CHECK(pos_got.size() == pos_ref.size());
  int64_t pos_bad = 0;
  for (size_t i = 0; i < pos_got.size() && i < pos_ref.size(); ++i) {
    if (pos_got[i] != pos_ref[i]) ++pos_bad;
  }
  CHECK(pos_bad == 0);

  // ---- position-embedding interpolation taps -------------------------------------------
  const auto idx_ref = ReadInt(golden, "interp_indices");
  const auto wt_ref = ReadFloat(golden, "interp_weights");
  const auto interp = BuildPosEmbedInterpolation(grids, kNumGridPerSide, kMergeSize);
  CHECK(interp.indices.size() == idx_ref.size());
  CHECK(interp.weights.size() == wt_ref.size());
  int64_t idx_bad = 0;
  double wt_max = 0.0;
  for (size_t i = 0; i < interp.indices.size() && i < idx_ref.size(); ++i) {
    if (interp.indices[i] != idx_ref[i]) ++idx_bad;
    wt_max = std::max(wt_max, std::fabs(static_cast<double>(interp.weights[i]) -
                                         static_cast<double>(wt_ref[i])));
  }
  CHECK(idx_bad == 0);
  CHECK(wt_max == 0.0);  // both sides are the same fp32 expression; anything nonzero is a real bug

  // ---- the position embedding itself ---------------------------------------------------
  const auto pe_ref = ReadFloat(golden, "pos_embeds");
  std::vector<float> pe_got(static_cast<size_t>(interp.num_patches) * kHiddenSize, 0.0f);
  for (int64_t p = 0; p < interp.num_patches; ++p) {
    for (int k = 0; k < 4; ++k) {
      const int32_t row = interp.indices[static_cast<size_t>(p) * 4 + k];
      const float w = interp.weights[static_cast<size_t>(p) * 4 + k];
      const float* src = pos_embed_table.data() + static_cast<size_t>(row) * kHiddenSize;
      float* dst = pe_got.data() + static_cast<size_t>(p) * kHiddenSize;
      for (int c = 0; c < kHiddenSize; ++c) dst[c] += src[c] * w;
    }
  }
  const ErrorStats pe = CompareFloat(pe_got, pe_ref);
  CHECK(pe.max_abs <= kPosEmbedAbsTol);

  // ---- vision rope cos/sin --------------------------------------------------------------
  std::vector<float> cos_got, sin_got;
  BuildVisionRopeCosSin(pos_got, kHeadDim, kVisionRopeTheta, &cos_got, &sin_got);
  const ErrorStats cos_err = CompareFloat(cos_got, ReadFloat(golden, "rope_cos"));
  const ErrorStats sin_err = CompareFloat(sin_got, ReadFloat(golden, "rope_sin"));
  CHECK(cos_err.max_abs <= kRopeAbsTol);
  CHECK(sin_err.max_abs <= kRopeAbsTol);

  std::printf("[test_vision_index] %-34s segments=%zu patches=%lld pos_embed_max_abs=%.3g "
              "cos_max_abs=%.3g sin_max_abs=%.3g\n",
              golden_file.c_str(), cu_got.size() - 1,
              static_cast<long long>(interp.num_patches), pe.max_abs, cos_err.max_abs,
              sin_err.max_abs);
}

}  // namespace

int main() {
  const std::vector<std::string> needed = {"vision_tower.safetensors",
                                            "vision_tower_nonsquare.safetensors",
                                            "vision_tower_two_image.safetensors"};
  for (const std::string& f : needed) {
    if (!FileExists(GoldenPath(f))) return SkipMissing(GoldenPath(f));
  }

  try {
    const r4dx_convert::SafetensorsReader base(
        r4dx_convert::Utf8ToWide(GoldenPath("vision_tower.safetensors")));
    const auto pos_embed_table = ReadFloat(base, "pos_embed_table");
    CHECK(pos_embed_table.size() ==
          static_cast<size_t>(kNumGridPerSide) * kNumGridPerSide * kHiddenSize);

    // inv_freq: head_dim/4 = 18 frequencies at theta=10000 -- the vision tower's own rope base,
    // not the text side's 1e7. A mix-up here is silent (both produce plausible cos/sin).
    const auto inv_freq_ref = ReadFloat(base, "rope_inv_freq");
    std::vector<int32_t> probe = {1, 0};  // position (h=1, w=0): cos row == cos(inv_freq)
    std::vector<float> cos_probe, sin_probe;
    BuildVisionRopeCosSin(probe, kHeadDim, kVisionRopeTheta, &cos_probe, &sin_probe);
    CHECK(inv_freq_ref.size() == static_cast<size_t>(kHeadDim) / 4);
    double inv_freq_max = 0.0;
    for (size_t i = 0; i < inv_freq_ref.size(); ++i) {
      inv_freq_max = std::max(inv_freq_max,
                               std::fabs(static_cast<double>(sin_probe[i]) -
                                          std::sin(static_cast<double>(inv_freq_ref[i]))));
    }
    CHECK(inv_freq_max <= kRopeAbsTol);

    for (const std::string& f : needed) CheckCase(f, pos_embed_table);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "test_vision_index: unexpected exception: %s\n", e.what());
    return 1;
  }

  if (g_failures != 0) {
    std::fprintf(stderr, "test_vision_index: %d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("test_vision_index: OK\n");
  return 0;
}
