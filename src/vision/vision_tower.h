// src/vision/vision_tower.h -- `Qwen3_5VisionModel.forward` on the GPU: patch embed, the learned
// position-embedding gather, 27 pre-norm encoder blocks over `r4d_attn_vit_h72_bf16`, and the 2x2
// patch merger that turns [num_patches, 1152] into the [num_merged_tokens, 5120] rows that get
// spliced into the text embedding sequence. docs/vision.md is the spec; this is the device half of
// it, on top of the host half (`preprocess.h` / `vision_index.h` / `position_ids.h`) that produces
// `pixel_values`, the interpolation taps, the rope positions and `cu_seqlens`.
//
// Everything is bf16 (`vision.*` is never quantized, docs/container-format.md), so every linear is
// one r4d_gemm_bf16_nt_m64 family call and there is exactly one numeric path to validate.
//
// SCOPE: one VisionTower serves one worker thread, like Model -- it owns a stream and a scratch
// arena that every Encode call reuses, so two concurrent Encode calls on the same object would
// interleave scratch.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "preprocess.h"  // GridThw
#include "r4dx/core/arena.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/stream.hpp"
#include "vision_weights.h"

namespace r4dx::vision {

// What one Encode call did, for the CLI/server log line and for docs/vision.md's measurements.
struct VisionEncodeStats {
  int64_t total_patches = 0;
  int64_t merged_tokens = 0;
  int64_t num_segments = 0;   // cu_seqlens entries - 1 (one per frame, so one per still image)
  int64_t max_seqlen = 0;     // the longest segment, which picks the attention kernel's query block
  int64_t row_chunk = 0;      // rows per scratch chunk this call used (see kDefaultRowChunk)
  int64_t scratch_bytes = 0;  // the arena's capacity after this call
  double encode_ms = 0.0;     // wall clock, H2D of pixel_values through the merger's last GEMM
};

// An optional per-tensor observer, used by tests/vision/test_vision_tower.cpp to diff every named
// intermediate against tools/reference/golden_out/vision_tower*.safetensors. Called with a HOST
// copy valid only for the duration of the call.
//
// Attaching a trace also forces the row chunk to `total_patches` for that call, so every traced
// intermediate exists whole rather than one chunk at a time. That changes buffer SIZES only: every
// op between the chunk boundaries is row-independent (GEMM rows, bias, activation, rope, residual
// add), and the GEMM itself is chunked to 64 rows internally either way, so a traced encode and an
// untraced one compute bit-identical results.
using VisionTrace =
    std::function<void(const std::string& name, const void* host_data, int64_t elems, bool is_fp32)>;

// A second diagnostic hook, called just before encoder block `b` runs with the DEVICE residual
// stream ([total_patches, hidden] bf16). A callback that overwrites it and returns true
// "teacher-forces" that block from the reference's own input -- which is the only way to separate
// a block's OWN numeric error from the error it inherited from every block before it, and is what
// tests/vision/test_vision_tower.cpp's per-block localization does. Returning false leaves the
// stream untouched. The encode stream is synchronized before the call, so the callback may use a
// blocking hipMemcpy.
using VisionPreBlock = std::function<bool(int64_t block, void* dev_x, int64_t elems)>;

// Holds only the encode stream and the scratch arena -- NOT the weights. Deliberately: the weights
// live inside the Container, and a Model (which owns both) is moved at least once on its way out of
// Model::Load, which would leave a cached `const VisionWeights*` pointing at the moved-from
// container. Passing the weights per call removes that whole class of bug rather than documenting
// around it.
class VisionTower {
 public:
  VisionTower() = default;

  // pixel_values: [total_patches, patch_dim] fp32, HOST, exactly what PreprocessImages produced --
  // rounded to bf16 on the way in, which is what the reference's own
  // `hidden_states.to(self.proj.weight.dtype)` does before the patch-embed matmul.
  // grids: one row per image, in the same order the patches were concatenated in.
  // out: resized to [sum(merged tokens), out_hidden_size] bf16 on the device.
  void Encode(const VisionWeights& weights, const float* pixel_values, int64_t total_patches,
              const std::vector<GridThw>& grids, core::DeviceBuffer<uint16_t>* out,
              VisionEncodeStats* stats = nullptr, const VisionTrace* trace = nullptr,
              const VisionPreBlock* pre_block = nullptr);

  // Device bytes the scratch arena currently holds (0 before the first Encode).
  int64_t ScratchBytes() const { return static_cast<int64_t>(arena_.capacity_bytes()); }

  // The encode stream, for a caller that must drain it after an Encode that threw partway (docs/tp.md
  // 2.5 step 1: TpModel's recovery, on the tower's rank thread).
  hipStream_t StreamHandle() const { return stream_.get(); }

  // The scratch an ordinary (untraced) encode of `total_patches` patches at `row_chunk` rows per
  // chunk needs -- the sizing a caller uses to choose --image-max-pixels, and exactly the number
  // docs/vision.md's "Large images" table reports. Does not allocate.
  static int64_t PlanScratchBytes(const VisionConfig& cfg, int64_t total_patches,
                                   int64_t row_chunk);

  // Rows per scratch chunk for the qkv / mlp / merger intermediates. 1024 caps the widest of them
  // (the MLP's [rows, 4304]) at ~8.8 MiB regardless of image size, which is what keeps a
  // 1536x1536 image's scratch at ~0.16 GiB instead of ~0.35 GiB. A multiple of 64 so it never
  // splits one of ApplyLinear-style 64-row GEMM sub-chunks.
  static constexpr int64_t kDefaultRowChunk = 1024;

 private:
  struct Scratch;
  void EnsureScratch(const VisionConfig& cfg, int64_t total_patches, int64_t row_chunk,
                     bool tracing);
  // y[M,N] = x[M,K] @ w.weight^T, in <=64-row GEMM launches (r4d_gemm_bf16_nt_m64's own M cap).
  void Gemm(const VisionLinear& w, const uint16_t* x, uint16_t* y, int64_t M);
  void Emit(const VisionTrace* trace, const std::string& name, const void* dev, int64_t elems,
            bool is_fp32);

  core::Stream stream_;
  core::Arena arena_;  // its capacity IS the "what was this sized for" state -- see EnsureScratch
  std::vector<uint8_t> trace_staging_;
};

}  // namespace r4dx::vision
