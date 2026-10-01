// r4dx::model::GemmaVisionWeights / GemmaVisionEmbed -- the device half of Gemma 4's encoder-free vision
// (docs/gemma4-plan.md M2; host half: src/vision/gemma_vision.{h,cpp}). No encoder: raw 48x48x3 merged pixel
// patches go through
//
//   LayerNorm(6912, eps 1e-5, w+b) -> Linear(6912 -> 3840, +bias) -> LayerNorm(3840)
//   -> + pos_embedding[x, 0] + pos_embedding[y, 1] -> LayerNorm(3840) ("pos_norm")
//   -> RMSNorm(3840, no weight, eps 1e-6) -> Linear(3840 -> 3840, no bias)
//
// and the result is spliced into the residual stream UNSCALED (text rows carry the sqrt(3840) embed scale,
// image rows do not). Everything is bf16 on the device (vision.* is never quantized): the two linears are
// ApplyLinear on a kBf16 QuantLinear, the norms / bias / position add reuse the existing r4dx kernels
// (r4dx_layernorm_bf16, r4dx_bias_add_bf16, r4dx_vision_pos_embed_bf16 with two taps of weight 1.0 over the
// table viewed as [2 * posemb, 3840] rows, r4dx_rmsnorm_noscale_bf16): no new kernel.
//
// Rounding: matches the bf16 torch module chain except the Linear bias, which is added after the GEMM's bf16
// rounding (one extra rounding, ~2^-9 relative; the same accepted deviation as the Qwen tower, kernels.h
// r4dx_bias_add_bf16).
//
// Container tensors (written by r4dx-convert --vision on; HF name with `model.` stripped):
//   vision.vision_embedder.{patch_ln1,patch_ln2,pos_norm}.{weight,bias}   bf16 [dim]
//   vision.vision_embedder.patch_dense.{weight [3840, 6912], bias [3840]}
//   vision.vision_embedder.pos_embedding                                  bf16 [1120, 2, 3840]
//   vision.embed_vision.embedding_projection.weight                       bf16 [3840, 3840]
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "model_types.h"
#include "quant_linear.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/stream.hpp"
#include "r4dx_convert/safetensors_reader.hpp"

namespace r4dx::model {

struct GemmaVisionWeights {
  int64_t patch_dim = 0;    // 6912
  int64_t mm_dim = 0;       // 3840
  int64_t out_dim = 0;      // text hidden, 3840
  int64_t posemb_size = 0;  // 1120
  float ln_eps = 1e-5f;
  float rms_eps = 1e-6f;
  QuantLinear dense;  // bf16 [mm_dim, patch_dim]
  QuantLinear proj;   // bf16 [out_dim, mm_dim]
  core::DeviceBuffer<uint16_t> ln1_w, ln1_b, dense_b, ln2_w, ln2_b, pos_norm_w, pos_norm_b;
  core::DeviceBuffer<uint16_t> pos_table;  // [posemb_size * 2, mm_dim] (the [posemb, 2, mm] table, flattened)
};

// True iff the container carries the vision embedder tensors (the projection weight is the probe).
bool GemmaContainerHasVisionTensors(const r4dx_convert::SafetensorsReader& r);

// Reads and shape-checks every vision tensor against `hidden` (the text hidden size). Throws
// std::runtime_error naming the container, the tensor and both shapes on a mismatch.
GemmaVisionWeights LoadGemmaVisionWeights(const r4dx_convert::SafetensorsReader& r, int64_t hidden,
                                          const std::string& path);

// Encodes `n` merged patches on `stream` and leaves the result in `out` ([n, out_dim] bf16, resized): the
// stream is synchronized on return. `pixels` is [n, patch_dim] fp32 on the host (rescaled to [0, 1], what
// PreprocessGemmaImage produced), rounded to bf16 on the way in as the reference's
// `pixel_values.to(weight.dtype)` does. `positions` is [n, 2] (x, y) merged-grid cells, all in
// [0, posemb_size). `arena` supplies the scratch and is Reset() on return.
void GemmaVisionEmbed(core::Stream& stream, core::Arena& arena, const GemmaVisionWeights& w, const float* pixels,
                      const int32_t* positions, int64_t n, core::DeviceBuffer<uint16_t>* out);

// Positions (x, y) of a [grid_h, grid_w] merged grid in raster order (the processor's order).
std::vector<int32_t> GemmaGridPositions(int64_t grid_h, int64_t grid_w);

}  // namespace r4dx::model
