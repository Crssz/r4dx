// r4dx::model -- the plain data types shared by r4dx::model::Model (model.h) and the TextModel
// interface src/cli and src/server call (text_model.h, docs/tp.md 2.8): ImageSpan, ImageRows,
// ProfileEntry and StepProfile. They live here, outside model.h, so text_model.h can use them
// without including model.h (no include cycle, and a TextModel caller does not see Model).
//
// model.h keeps `using ImageSpan = r4dx::model::ImageSpan;` (and the same for ProfileEntry /
// StepProfile) inside class Model, so every existing `Model::ImageSpan` / `Model::StepProfile`
// spelling still compiles and names the same type.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "preprocess.h"  // src/vision: vision::GridThw
#include "r4dx/core/device_buffer.hpp"

namespace r4dx::model {

// One image occurrence inside a PrefillMultimodal call's OWN token vector (docs/vision.md
// "Text-side splicing").
struct ImageSpan {
  // Index of the first image-placeholder token (config.json's image_token_id, 248056) within
  // THIS call's `token_ids` -- not an absolute sequence index, so a continuation prefill states
  // its spans relative to the tail it is feeding, exactly as it states its tokens.
  int64_t offset = 0;
  // Length of the placeholder run == the image's MERGED token count
  // (grid.MergedTokenCount(merge_size)). Validated against `grid` and against the tokens
  // actually present at [offset, offset+tokens).
  int64_t tokens = 0;
  // The image's PATCH grid (pre-merge), which is what the mrope advance rule
  // `current_pos += max(grid.h, grid.w) / merge_size` reads -- NOT the merged grid, and NOT the
  // token count (docs/vision.md).
  vision::GridThw grid;
  // [tokens, hidden] bf16: this image's rows of the EncodeImages output, which overwrite the
  // embed_tokens lookup at the placeholder positions. Device memory unless `embeds_on_host`.
  const uint16_t* embeds = nullptr;
  // docs/tp.md 8.3: true => `embeds` is HOST memory (a tensor-parallel ImageRows), spliced with an
  // H2D copy on every rank. Always false at TP=1 (device rows, the D2D splice) -- and, until
  // docs/tp.md P5 enables vision under TP, always false everywhere.
  bool embeds_on_host = false;
};

// Owner of one EncodeImages result (docs/tp.md 2.8). TP=1: device rows in `dev` (exactly the
// DeviceBuffer Model::EncodeImages writes). TP: pageable host rows in `host`, copied to each rank's
// device at splice time (docs/tp.md 8.3) -- pageable so that releasing it on the facade thread is
// not a HIP call.
class ImageRows {
 public:
  const uint16_t* data() const { return on_host_ ? host.data() : dev.data(); }
  bool on_host() const { return on_host_; }
  int64_t rows() const { return rows_; }
  // Called by whoever filled `dev` or `host`: which one holds the rows, and how many [hidden]-wide
  // rows there are.
  void SetFilled(bool on_host, int64_t rows) {
    on_host_ = on_host;
    rows_ = rows;
  }

  core::DeviceBuffer<uint16_t> dev;  // filled by LocalTextModel
  std::vector<uint16_t> host;        // filled by TpModel (docs/tp.md P5)

 private:
  bool on_host_ = false;
  int64_t rows_ = 0;
};

// One named GPU timing span's accumulated result for one profiled decode step (tools/profile
// pass, 2026-09-19): `count` calls of this op family summed to `ms` milliseconds of hipEvent-
// measured device time (GDN kernels are one entry per GDN layer's whole Forward(), attention
// likewise, one entry per distinct GEMM (N,K) shape, etc. -- see DecodeStepProfiled's own
// comment for exactly what is and is not broken out).
struct ProfileEntry {
  std::string name;
  double ms = 0.0;
  int count = 0;
};
// NOTE on what NOT to do with these numbers: `entries[*].ms` (hipEvent-measured) and
// `finish_wait_ms` (host-chrono-measured) are NOT additive -- every kernel `entries` accounts
// for was already enqueued (asynchronously) before `finish_wait_ms`'s clock starts, so
// `finish_wait_ms` is host time spent BLOCKED waiting for that SAME already-queued GPU work to
// finish (plus the final 4-byte D2H), not extra work on top of it. `entries[*].ms` summed
// (`gpu_sum_ms`) is the right number for "which kernel family dominates GPU time" (this
// pass's own top-3-cost ask); `finish_wait_ms` cross-validates it from the host's point of view
// (the two should land close to each other) and is also the honest place to see the "only sync
// per token" cost the host-overhead pass (item 5) was trying to minimize. `wall_ms` ~=
// `host_enqueue_ms` (the host-only time spent issuing every async kernel launch + CPU-side work
// like the embedding gather, measured BEFORE the Finish()/sync call below) + `finish_wait_ms`.
struct StepProfile {
  std::vector<ProfileEntry> entries;  // per op-family, hipEvent-measured GPU device time, in
                                       // call order (first occurrence) -- does NOT include the
                                       // final stream-sync+readback (see finish_wait_ms)
  double gpu_sum_ms = 0.0;            // sum of entries[*].ms
  double host_enqueue_ms = 0.0;       // host time to issue every async launch (wall_t0 up to
                                       // the Finish()/sync call below), NOT GPU-blocked
  double finish_wait_ms = 0.0;        // host-chrono time for Finish()'s hipEventSynchronize +
                                       // the final 4-byte argmax D2H -- the actual "only sync
                                       // per token" cost, see NOTE above
  double wall_ms = 0.0;               // host wall-clock for the whole DecodeStepProfiled call
                                       // (~= host_enqueue_ms + finish_wait_ms)
  int64_t r4dx_kernel_launches = 0;   // r4dx::kernels::r4dx_kernel_launch_counter_get() delta for
                                       // just this one step (docs/r9700.md P2/task item 4) --
                                       // r4dx-owned launches only, NOT third_party/libr4d's own
                                       // r4d_gemm_*/r4d_gdn_*/r4d_attn_* launches; see kernels.h.
};

}  // namespace r4dx::model
