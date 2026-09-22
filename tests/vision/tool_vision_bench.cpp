// tests/vision/tool_vision_bench.cpp -- vision-tower encode cost and VRAM high-water mark, next to
// the real 27B text model (docs/vision.md "Large images").
//
// Built but deliberately NOT registered with add_test(), the same convention tests/model's
// tool_hseed_drift / tool_vocab_calib and tests/kernels' tool_sampler_bench follow: it prints
// numbers for a human and for docs/vision.md, it asserts no pass/fail contract, so it must never
// move the ctest count.
//
//   $env:HIP_VISIBLE_DEVICES='1'
//   build\win-hip\tests\vision\tool_vision_bench.exe [--model D:/models/r4dx/qwen38-27b-v6.r4dx]
//       --layout w4a16 [--sizes 448,1024,1536] [--no-model] [--image-max-pixels N] [--runs 3]
//
// `--no-model` skips loading the text model and measures the tower alone; the default loads the
// real 27B first, which is the configuration the "does a 1536x1536 image fit next to the model"
// question is actually about.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "../model/test_container_path.h"  // r4dx_test::ProductionTargetPath
#include "model.h"
#include "preprocess.h"
#include "vision_tower.h"
#include "vision_weights.h"

namespace {

double GiB(int64_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0); }

struct Vram {
  int64_t free_bytes = 0, total_bytes = 0;
};
Vram SnapVram() {
  size_t f = 0, t = 0;
  Vram v;
  if (hipMemGetInfo(&f, &t) == hipSuccess) {
    v.free_bytes = static_cast<int64_t>(f);
    v.total_bytes = static_cast<int64_t>(t);
  }
  return v;
}

// A deterministic non-flat RGB image. Content does not change the encode's cost -- every kernel is
// shape-driven -- but a flat image would make a preprocessing mistake invisible if this tool is
// ever repurposed as a sanity check.
r4dx::vision::DecodedImage MakeImage(int w, int h) {
  r4dx::vision::DecodedImage img;
  img.width = w;
  img.height = h;
  img.rgb.resize(static_cast<size_t>(w) * h * 3);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const size_t o = (static_cast<size_t>(y) * w + x) * 3;
      img.rgb[o + 0] = static_cast<uint8_t>((x * 255) / std::max(1, w - 1));
      img.rgb[o + 1] = static_cast<uint8_t>((y * 255) / std::max(1, h - 1));
      img.rgb[o + 2] = static_cast<uint8_t>(((x / 16) + (y / 16)) % 2 ? 200 : 40);
    }
  }
  return img;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path = r4dx_test::ProductionTargetPath();  // group-matched v6 (64) / v3 (128)
  std::string layout = "w4a16";
  std::string sizes = "448,1024,1536";
  int64_t layers = -1, max_ctx = 2048, image_max_pixels = 0, runs = 3;
  bool load_model = true;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (a == "--model") model_path = next();
    else if (a == "--layout") layout = next();
    else if (a == "--sizes") sizes = next();
    else if (a == "--layers") layers = std::stoll(next());
    else if (a == "--max-ctx") max_ctx = std::stoll(next());
    else if (a == "--image-max-pixels") image_max_pixels = std::stoll(next());
    else if (a == "--runs") runs = std::stoll(next());
    else if (a == "--no-model") load_model = false;
    else {
      std::fprintf(stderr, "unrecognized argument: %s\n", a.c_str());
      return 2;
    }
  }

  const Vram v0 = SnapVram();
  std::printf("[bench] VRAM at start: %.3f / %.3f GiB free\n", GiB(v0.free_bytes),
              GiB(v0.total_bytes));

  r4dx::model::ModelOptions opts;
  opts.container_path = model_path;
  opts.layout = r4dx::model::LayoutFromName(layout);
  opts.layer_limit = layers;
  opts.max_ctx = max_ctx;
  opts.vision = r4dx::model::ModelOptions::VisionMode::kOn;

  std::optional<r4dx::model::Model> model;
  r4dx::vision::VisionWeights standalone_weights;
  const r4dx::vision::VisionWeights* weights = nullptr;
  r4dx::vision::VisionTower standalone_tower;

  if (load_model) {
    model.emplace(r4dx::model::Model::Load(opts));
    weights = &model->GetContainer().Vision();
  } else {
    const std::string wide_src = model_path;
    r4dx_convert::SafetensorsReader reader(r4dx_convert::Utf8ToWide(wide_src));
    // The container's metadata block, re-parsed the same way container.cpp's ReadMetadata does.
    std::vector<char> header;
    {
      std::ifstream f(model_path, std::ios::binary);
      if (!f) {
        std::fprintf(stderr, "cannot open %s\n", model_path.c_str());
        return 1;
      }
      uint64_t len = 0;
      f.read(reinterpret_cast<char*>(&len), 8);
      header.resize(static_cast<size_t>(len));
      f.read(header.data(), static_cast<std::streamsize>(header.size()));
      if (!f) {
        std::fprintf(stderr, "truncated header in %s\n", model_path.c_str());
        return 1;
      }
    }
    const nlohmann::json meta =
        nlohmann::json::parse(std::string(header.begin(), header.end())).at("__metadata__");
    standalone_weights = r4dx::vision::LoadVisionWeights(
        reader, meta.at("model_config").at("vision_config"));
    weights = &standalone_weights;
  }
  const Vram v1 = SnapVram();
  std::printf("[bench] VRAM after load: %.3f GiB free (this load consumed %.3f GiB; the vision "
              "tower's own tensors sum to %.4f GiB)\n",
              GiB(v1.free_bytes), GiB(v0.free_bytes - v1.free_bytes), GiB(weights->bytes));

  r4dx::vision::ImageProcessorConfig pcfg;
  if (image_max_pixels > 0) pcfg.max_pixels = image_max_pixels;

  std::printf("\n%-10s %-12s %8s %8s %10s %12s %12s %12s\n", "size", "grid", "patches", "merged",
              "encode_ms", "scratch_MiB", "free_GiB", "peak_used_GiB");
  int64_t low_water = v1.free_bytes;
  for (size_t pos = 0, next = 0; pos <= sizes.size(); pos = next + 1) {
    next = sizes.find(',', pos);
    if (next == std::string::npos) next = sizes.size();
    const std::string tok = sizes.substr(pos, next - pos);
    if (tok.empty()) break;
    const int side = std::stoi(tok);

    const r4dx::vision::DecodedImage img = MakeImage(side, side);
    const r4dx::vision::PreprocessedImages pre =
        r4dx::vision::PreprocessImages({img}, pcfg);
    const int64_t patches = pre.TotalPatches();

    r4dx::core::DeviceBuffer<uint16_t> out;
    r4dx::vision::VisionEncodeStats stats;
    double best = 1e30;
    for (int64_t r = 0; r < runs + 1; ++r) {  // run 0 is a discarded warm-up
      if (model) {
        model->EncodeImages(pre.pixel_values.data(), patches, pre.grid_thw, &out, &stats);
      } else {
        standalone_tower.Encode(*weights, pre.pixel_values.data(), patches, pre.grid_thw, &out,
                                 &stats);
      }
      if (r > 0) best = std::min(best, stats.encode_ms);
    }
    const Vram v = SnapVram();
    low_water = std::min(low_water, v.free_bytes);
    char grid[24];
    std::snprintf(grid, sizeof(grid), "%lldx%lld", static_cast<long long>(pre.grid_thw[0].h),
                  static_cast<long long>(pre.grid_thw[0].w));
    std::printf("%-10d %-12s %8lld %8lld %10.1f %12.1f %12.3f %12.3f\n", side, grid,
                static_cast<long long>(patches),
                static_cast<long long>(stats.merged_tokens), best,
                static_cast<double>(stats.scratch_bytes) / (1024.0 * 1024.0),
                GiB(v.free_bytes), GiB(v0.free_bytes - v.free_bytes));
  }
  std::printf("\n[bench] VRAM high-water: %.3f GiB free at the tightest point (%.3f GiB used in "
              "total from this process's start)\n",
              GiB(low_water), GiB(v0.free_bytes - low_water));
  return 0;
}
