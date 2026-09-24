#include "local_text_model.h"

#include <hip/hip_runtime.h>

#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/error.hpp"

namespace r4dx::model {

int64_t LocalTextModel::VisionMergeSize() const {
  // Exactly what src/cli and src/server read before TextModel existed:
  // GetContainer().Vision().config.spatial_merge_size when the tower is loaded, else 2.
  return m_.HasVision() ? static_cast<int64_t>(m_.GetContainer().Vision().config.spatial_merge_size) : 2;
}

std::vector<VramReport> LocalTextModel::Vram() const {
  VramReport r;
  r.rank = 0;
  R4DX_HIP_CHECK(hipGetDevice(&r.device));
  size_t free_b = 0, total_b = 0;
  R4DX_HIP_CHECK(hipMemGetInfo(&free_b, &total_b));
  constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
  r.used_gib = static_cast<double>(total_b - free_b) / kGiB;
  r.free_gib = static_cast<double>(free_b) / kGiB;
  r.total_gib = static_cast<double>(total_b) / kGiB;
  r.buffers_gib = static_cast<double>(core::DeviceBufferBytes(r.device)) / kGiB;
  return {r};
}

void LocalTextModel::EncodeImages(const float* pixel_values, int64_t total_patches,
                                  const std::vector<vision::GridThw>& grids, ImageRows* out,
                                  vision::VisionEncodeStats* stats) {
  m_.EncodeImages(pixel_values, total_patches, grids, &out->dev, stats);
  int64_t rows = 0;
  const int64_t merge = VisionMergeSize();
  for (const vision::GridThw& g : grids) rows += g.MergedTokenCount(merge);
  out->SetFilled(/*on_host=*/false, rows);
}

}  // namespace r4dx::model
